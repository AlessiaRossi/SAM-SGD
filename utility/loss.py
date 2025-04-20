import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.log import Log
import torch.optim as optim


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # classifica multi-classe
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LogitNormLoss(nn.Module):
    def __init__(self, device, t=1.0):
        super(LogitNormLoss, self).__init__()
        self.device = device
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)


def squared_l2_norm(x):
    flattened = x.view(x.size(0), -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


class TRADESLoss(nn.Module):
    def __init__(self, model, optimizer, step_size, epsilon, perturb_steps, beta, distance):
        super(TRADESLoss, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta
        self.distance = distance

        # Inizializza KLDivLoss
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, inputs, targets):
        """
        Calcola la perdita TRADES.
        """
        # Genera perturbazioni avversarie
        x_adv = inputs.detach() + 0.001 * torch.randn_like(inputs).detach()
        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_adv = F.cross_entropy(self.model(x_adv), targets)
            grad = torch.autograd.grad(loss_adv, [x_adv])[0]

            # Calcola la norma dei gradienti
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True)
            grad_norm = torch.clamp(grad_norm, min=1e-8)  # Evita divisioni per zero

            # Clipping dinamico dei gradienti
            grad = grad / grad_norm  # Normalizza i gradienti
            if self.distance == "l_inf":
                x_adv = x_adv + self.step_size * grad.sign()
                x_adv = torch.clamp(x_adv, inputs - self.epsilon, inputs + self.epsilon)
            elif self.distance == "l_2":
                x_adv = x_adv + self.step_size * grad
                delta = x_adv - inputs
                delta_norm = torch.norm(delta.view(delta.size(0), -1), dim=1, keepdim=True)
                mask = delta_norm > self.epsilon
                delta[mask] = self.epsilon * delta[mask] / delta_norm[mask]
                x_adv = inputs + delta
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        # Calcola la perdita TRADES
        logits = self.model(inputs)
        logits_adv = self.model(x_adv)
        logits = logits - logits.max(dim=1, keepdim=True).values  # Normalizzazione
        logits_adv = logits_adv - logits_adv.max(dim=1, keepdim=True).values
        loss_ce = F.cross_entropy(logits, targets)
        loss_kl = self.kl_div(F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1))
        loss = loss_ce + self.beta * loss_kl
        return loss


class HuberLoss(nn.Module):
    """
    Implementazione della Huber Loss per la classificazione.
    """
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, inputs, targets):
        # Calcola la Cross Entropy Loss per i logits
        loss = F.cross_entropy(inputs, targets, reduction='none')

        # Calcola i residui tra logits e target (one-hot encoded)
        residual = F.one_hot(targets, num_classes=inputs.size(1)).float() - F.softmax(inputs, dim=1)
        residual = residual.sum(dim=1)  # Somma lungo la dimensione delle classi

        # Applica la Huber Loss
        condition = residual.abs() < self.delta
        huber_loss = torch.where(condition, 0.5 * residual**2, self.delta * (residual.abs() - 0.5 * self.delta))

        return huber_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, loss1, loss2, lambda_):
        super(CombinedLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.lambda_ = lambda_

    def _compute_loss(self, loss_fn, model, inputs, targets):
        # Se accetta 3 argomenti: (model, inputs, targets)
        if isinstance(loss_fn, TRADESLoss):
            return loss_fn(inputs, targets)
        else:
            logits = model(inputs)
            return loss_fn(logits, targets)

    def forward(self, model, inputs, targets):
        loss1_value = self._compute_loss(self.loss1, model, inputs, targets)
        loss2_value = self._compute_loss(self.loss2, model, inputs, targets)
        return self.lambda_ * loss1_value + (1 - self.lambda_) * loss2_value
