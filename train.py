import sys
import os

from model import UNet
from eval import eval_net
from util import *

import torch
import torch.nn as nn
from torch import optim
from optparse import OptionParser

def train_net(net,
              epochs = 5,
              batch_size = 1,
              lr = 0.1,
              val_percent = 0.1,
              save_cp = False,
              gpu = True):
    
    img_dir = 'images/'
    mask_dir = 'masks/'
    checkpoint_dir ='checkpoints/'
    
    name_list = get_names(image_dir)
    split_list = train_val_split(name_list, val_percent)
    
    print('''
        Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(split_list['train']),
               len(split_list['val']), str(save_cp), str(gpu)))
    
    N_train = len(split_list['train'])
    optimizer = optim.Adam(net.parameters(), 
                            lr=lr, 
                            momentum = 0.9,
                            weight_decay=0.005)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        
        val = get_val_pics(image_dir, mask_dir, split_list)
        train = get_train_pics(image_dir, mask_dir, split_list)
        
        epoch_loss = 0
        
        for i, samps in enumerate(batch(val, batch_size)):
            images = np.array([samp['image'] for samp in samps])
            masks = np.array([samp['mask'] for samp in samps])
    
            images = torch.from_numpy(images)
            masks = torch.from_numpy(masks)
            
            if gpu:
                images = images.cuda()
                true_masks = masks.cuda()
            
            masks_pred = net(images)              
            
            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat = true_masks.view(-1)
            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()
            
#                print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            print('Epoch finished ! Loss: {}'.format(epoch_loss / i))
    
def get_args():
    parser = OptionParser()
    parser.add_option('-e','--epochs',dest = 'epochs', default = 5, type = 'int',
                      help = 'number of epochs') 
    parser.add_option('-b','--batch-size', dest = 'batchsize', default = 5, type = 'int',
                      help = 'batchsize') 
    parser.add_option('-l', '--learning-rate', dest = 'lr', default = 0.1, type = float,
                      help = 'learning rate')
    parser.add_option('-g', '--gpu', dest = 'gpu', action = 'store_true', default = True, 
                      help = 'use cuda') 
    parser.add_option('-c', '--load', dest = 'load', default = False,
                      help = 'load file model')
    parser.add_option('-s', '--scale', dest = 'scale', default = 1, type = float,
                      help = 'downscaling factor of the images') 
    
    (options, args) = parser.parse_args() 
    return options          