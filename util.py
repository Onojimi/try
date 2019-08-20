import os
import torch
import numpy as np
from PIL import Image

def normalize(x):
    return x/255
 
 
def get_names(dir_img):
    return [f[:-4] for f in os.listdir(dir_img)]


def train_val_split(list_data, percent):
    split_point = int(len(list_data)*percent)
    val = list_data[:split_point]
    train = list_data[split_point:]
    return {'train':train, 'val':val}


def get_val_pics(dir_img, dir_mask, split_data):
    for b in split_data['val']:
        img = np.array(Image.open(dir_img + b + '.png').convert('RGB'))
        img = normalize(np.transpose(img, [2,0,1]))
        
        mask = np.array(Image.open(dir_mask + b + '.png').convert('L'))
        mask = normalize(mask)
        
        yield {'image':img, 'mask' : mask}


def get_train_pics(dir_img, dir_mask, split_data):
    for b in split_data['train']:
        img = np.array(Image.open(dir_img + b + '.png').convert('RGB'))
        img = normalize(np.transpose(img, [2,0,1]))
        
        mask = np.array(Image.open(dir_mask + b + '.png').convert('L'))
        mask = normalize(mask)
        
        yield {'image':img, 'mask' : mask}


def batch(itera, batch_size):     
    
    bat = [] 
      
    for i, samp in enumerate(itera):
        bat.append(samp)
        if (i+1) % batch_size == 0:
            yield bat
            bat = []
    
    if len(bat) > 0:
        yield bat
 
# img = Image.open('test.png')   
# img_np = np.array(img)
# img_np = img_np[:,:,:3]
# img = Image.fromarray(img_np)
# img.save('test.png')
# image_dir = 'images/'
# mask_dir = 'masks/'
# 
# name_list = get_names(image_dir)
# split_list = train_val_split(name_list, 0.1)
# 
# val = get_val_pics(image_dir, mask_dir, split_list)
# train = get_train_pics(image_dir, mask_dir, split_list)
# 
# for i, samps in enumerate(val):
#     print('x')
#     images = np.array(samps['image'])
#     masks = np.array(samps['mask'])
#      
#     images = torch.from_numpy(images)
#     masks = torch.from_numpy(masks)
#     