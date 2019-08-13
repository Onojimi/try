import torch
import torch.nn as nn
import numpy as np

def compute_iou(true, pred):
    true_mask = np.asanyarray(true, dtype = np.bool)
    pred_mask = np.asanyarray(pred, dtype = np.bool)
    union = np.sum(np.logical_or(true_mask, pred_mask))
    intersection = np.sum(np.logical_and(true_mask, pred_mask))
    iou = intersection/union
    return iou


def eval_net(net, dataset, gpu = False):
    net.eval()
    iou = 0
    ls = 0
    ct = 0
    print(type(dataset))
    for i, samp in enumerate(dataset):
        image = np.array(samp['image'])
        mask = np.array(samp['mask'])
        
        image = torch.from_numpy(image).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        
        image = image.unsqueeze(0)
        
        if gpu:
            image = image.cuda()
            mask = mask.cuda()

        criterion = nn.BCELoss()
        
        mask_pred = net(image)[0]
       
        mask_pred = (mask_pred > 0.5).float()
        mask_pred_np = np.array(mask_pred.cpu()).squeeze(0)
        
        ls+= criterion(mask_pred.view(-1),mask.view(-1)) 

        iou += compute_iou(mask,mask_pred_np)
        ct+=1
    
    return iou/ct, ls/ct