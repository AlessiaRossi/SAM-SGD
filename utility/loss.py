import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.log import Log
from utility.lr import StepLR


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probabilità predetta
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CE_FL_Loss(nn.Module):
    def __init__(self, lambda_=1.0):
        super(CE_FL_Loss, self).__init__()
        self.lambda_ = lambda_
        self.cross_entropy = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss()

    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)
        return self.lambda_ * ce_loss + (1 - self.lambda_) * focal_loss


class LambdaOptimizer:
    """
    Classe per eseguire il training e visualizzare le metriche al variare di lambda.
    """
    def __init__(self, train_fn, dataset, model, optimizer, scheduler, args, step=0.2):
        self.train_fn = train_fn
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.step = step

    def run(self):
        # Loop su valori di lambda nell'intervallo [0, 1] con step definito
        for lambda_ in [round(i * self.step, 2) for i in range(int(1 / self.step) + 1)]:
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