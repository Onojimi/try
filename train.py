import sys
import os

from model import UNet
from eval import eval_net
from util import get_names, train_val_split, get_val_pics, get_train_pics, batch
                    
from combineloss import mixloss

import torch
import torch.nn as nn
import numpy as np
from torch import optim
from optparse import OptionParser
from tensorboardX import SummaryWriter

def train_net(net,
              writer,
              load,
              epochs = 5,
              batch_size = 1,
              lr = 0.1,
              val_percent = 0.1,
              save_cp = False,
              gpu = True,
              ):
    
    image_dir = 'images_cut/'
    mask_dir = 'masks_cut/'
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
                            weight_decay=0.005)
    if load:
        print('Model loaded from {}'.format(args.load))
        model_dict = net.state_dict()
        pretrained_dict = torch.load('CP50.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        train_params = []
        for k, v in net.named_parameters():
            train_params.append(k)
            pref = k[:12]
            if pref == 'module.conv1' or pref == 'module.conv2' :
                v.requires_grad=False
                train_params.remove(k)
        
        optimizer = optim.Adam(params=train_params,
                               lr=lr,
                               weight_decay=0.005)    
    
    criterion = mixloss()
    
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        
        train = get_train_pics(image_dir, mask_dir, split_list)
        
        epoch_loss = 0
        
        for i, samps in enumerate(batch(train, batch_size)):
            images = np.array([samp['image'] for samp in samps])
            masks = np.array([samp['mask'] for samp in samps])
    
            images = torch.from_numpy(images).type(torch.FloatTensor)
            masks = torch.from_numpy(masks).type(torch.FloatTensor)
            
            if gpu:
                images = images.cuda()
                true_masks = masks.cuda()
            
            masks_pred = net(images)              
            
            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat = true_masks.view(-1)
            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()
            
            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = epoch_loss / i
        print('Epoch finished ! Loss: {}'.format(avg_train_loss ))
        
        val = get_val_pics(image_dir, mask_dir, split_list)
            
        if 1:
            val_iou, val_ls = eval_net(net, val, gpu)
            print('Validation IoU: {} Loss:{}'.format(val_iou,val_ls))
        
        writer.add_scalar('train/loss', avg_train_loss, epoch)
        writer.add_scalar('val/loss', val_ls, epoch )
        writer.add_scalar('val/IoU', val_iou, epoch )

        torch.save(net.state_dict(),
        checkpoint_dir + 'CP{}.pth'.format(epoch + 1))
        print('Checkpoint {} saved !'.format(epoch + 1))
    
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

if __name__ == '__main__':
    args = get_args()
#     os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    net = UNet(input_channels=3, nclasses=1)
    writer = SummaryWriter(log_dir='../../log/color2', comment='unet')
#     net.cuda()
#     import pdb
#     from torchsummary import summary 
#     summary(net, (3,1000,1000))
#     pdb.set_trace()
   
        
    if args.gpu:
        if torch.cuda.device_count()>1:
            net = nn.DataParallel(net)
        net.cuda()
        
    try:
        train_net(net = net, 
                  epochs = args.epochs,
                  batch_size = args.batchsize, 
                  lr = args.lr, 
                  gpu = args.gpu, 
                  writer = writer,
                  load = args.load
                  )
        
        torch.save(net.state_dict(),'model_fin.pth')
        
        
    
    except KeyboardInterrupt:
        torch.save(net.state_dict(),'interrupt.pth')
        print('saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)         