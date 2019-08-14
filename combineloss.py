import torch
import torch.nn as nn
import torch.nn.functional as F

class mixloss(nn.Module):
    def __init__(self):
        super(mixloss, self).__init__()
        self.eps = 0.01

    def forward(self, pred_mask, true_mask):
        inter = torch.dot(pred_mask.view(-1), true_mask.view(-1))
        union = torch.sum(pred_mask) + torch.sum(true_mask) + self.eps
        
        dice_loss = torch.zeros(0, dtype = torch.float32)
        dice_loss += (2 * inter.float() + self.eps) / union.float()
        bce_loss = nn.BCELoss(pred_mask, true_mask)
        
        mix_loss = bce_loss + (1 - dice_loss)
        return mix_loss
        
        

