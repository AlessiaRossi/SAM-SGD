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
    def __init__(self, model, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, distance='l_inf'):
        super(TRADESLoss, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta
        self.distance = distance
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x_natural, y):
        batch_size = len(x_natural)
        self.model.eval()
        x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).cuda().detach()

        if self.distance == 'l_inf':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = self.kl_div(F.log_softmax(self.model(x_adv), dim=1),
                                           F.softmax(self.model(x_natural), dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        elif self.distance == 'l_2':
            delta = 0.001 * torch.randn_like(x_natural).cuda().detach()
            delta.requires_grad_()
            optimizer_delta = optim.SGD([delta], lr=self.epsilon / self.perturb_steps * 2)

            for _ in range(self.perturb_steps):
                adv = x_natural + delta
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    loss = -self.kl_div(F.log_softmax(self.model(adv), dim=1),
                                        F.softmax(self.model(x_natural), dim=1))
                loss.backward()
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                optimizer_delta.step()
                delta.data.add_(x_natural)
                delta.data.clamp_(0, 1).sub_(x_natural)
                delta.data.renorm_(p=2, dim=0, maxnorm=self.epsilon)
            x_adv = (x_natural + delta).detach()
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        self.model.train()
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
        self.optimizer.zero_grad()
        loss_natural = F.cross_entropy(self.model(x_natural), y)
        loss_robust = self.kl_div(F.log_softmax(self.model(x_adv), dim=1),
                                  F.softmax(self.model(x_natural), dim=1)) / batch_size
        return loss_natural + self.beta * loss_robust


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
        self.lambda_ = float(lambda_)

    def forward(self, model, inputs, targets):
        """
        Calcola la loss combinata tra loss1 e loss2.

        Args:
            model (torch.nn.Module): Il modello da addestrare.
            inputs (torch.Tensor): Gli input del modello.
            targets (torch.Tensor): I target associati agli input.

        Returns:
            torch.Tensor: La loss combinata.
        """
        # Ottieni i logits chiamando il modello con gli input
        logits = model(inputs)

        # Calcola la prima loss
        loss1_value = self.loss1(logits, targets)

        # Calcola la seconda loss
        if isinstance(self.loss2, TRADESLoss):
            loss2_value = self.loss2(model, inputs, targets)
        else:
            loss2_value = self.loss2(logits, targets)

        # Calcola la loss combinata
        combined_loss = self.lambda_ * loss1_value + (1 - self.lambda_) * loss2_value

        return combined_loss