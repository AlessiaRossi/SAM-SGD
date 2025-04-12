import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.log import Log
from utility.lr import StepLR
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


class CombinedLoss(nn.Module):
    def __init__(self, loss1: nn.Module, loss2: nn.Module, lambda_: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.lambda_ = lambda_

    def forward(self, inputs, targets):
        return self.lambda_ * self.loss1(inputs, targets) + (1 - self.lambda_) * self.loss2(inputs, targets)


class LambdaOptimizer:
    """
    Classe per eseguire il training e visualizzare le metriche al variare di lambda.
    """
    def __init__(self, train_fn, dataset, model, optimizer, scheduler, args, lambda_start=0.0, lambda_end=1.0, lambda_step=0.2):
        self.train_fn = train_fn
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.lambda_step = lambda_step

    def run(self):
        # Loop su valori di lambda nell'intervallo definito
        lambda_values = [round(i, 2) for i in torch.arange(self.lambda_start, self.lambda_end + self.lambda_step, self.lambda_step).tolist()]
        for lambda_ in lambda_values:
            print(f"\n>>> Testing lambda = {lambda_}")

            # Inizializza il logger per il valore corrente di lambda
            log = Log(
                log_each=10,
                log_file=f"training_log_lambda_{lambda_:.2f}.csv",
                model_name="model.pth",
                lambda_value=lambda_,
                optimize_lambda=True
            )

            # Esegui il training con il valore corrente di lambda
            self.train_fn(self.model, self.optimizer, self.scheduler, self.dataset, self.args, log, use_sam=False, lambda_optimizer=None)

            # Stampa tutte le metriche per il valore corrente di lambda
            print(f"Metrics for Lambda = {lambda_}:")
            log.print_best_metrics()  # Chiamata diretta al metodo di stampa delle metriche