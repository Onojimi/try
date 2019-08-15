import torch
import torch.nn as nn
import numpy as np
from torch import optim
from model import UNet
from torchvision import models

model = UNet(3,1)
model = nn.DataParallel(model)
model_dict = model.state_dict()
pretrained_dict = torch.load('CP50.pth')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

model_dict.update(pretrained_dict)

model.load_state_dict(model_dict)
train_params = []
for k, v in model.named_parameters():
    train_params.append(k)
    pref = k[:12]
    print(pref)
    if pref == 'module.conv1' or pref == 'module.conv2' :
        v.requires_grad=False
        train_params.remove(k)

for k, v in model.named_parameters():
    if k not in train_params:
        print(v.requires_grad)    