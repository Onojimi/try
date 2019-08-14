import torch
import torch.nn as nn
import torch.nn.functional as F

class mixloss(nn.Module):
    def __init__(self, pred_mask, true_mask):
        super(mixloss, self).__init__()
        self.eps = 0.01
        self.pred_mask = pred_mask
        self.true_mask = true_mask
        self.inter = torch.dot(pred_mask.view(-1), true_mask.view(-1))
        self.union = torch.sum(pred_mask) + torch.sum(true_mask) + self.eps
    def forward(self):
        dice_loss = torch.zeros(0, requires_grad = True, dtype = torch.float32)
        dice_loss += (2 * self.inter.float() + self.eps) / self.union.float()
        bce_loss = nn.BCELoss(self.pred_mask, self.true_mask)
        mix_loss = bce_loss + (1 - dice_loss)
        return mix_loss
        
        

