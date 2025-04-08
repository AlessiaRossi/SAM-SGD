import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, lambda_=0.5):
        super(CE_FL_Loss, self).__init__()
        self.lambda_ = lambda_
        self.cross_entropy = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss()

    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)
        return self.lambda_ * ce_loss + (1 - self.lambda_) * focal_loss