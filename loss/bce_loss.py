import torch
import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-7
        
    def forward(self, predict, label):
        label = label.detach()
        predict = torch.clamp(predict, self.eps, 1-self.eps) 
        bce = -((1-label)*(1-predict).log()).sum()/(1-label).sum() - (label*predict.log()).sum()/label.sum()
        return bce