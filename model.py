import torch
import torch.nn as nn

def conv_twice(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True),
        nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True)
    )

def down_pooling():
    return nn.MaxPool2d(2)

def up_pooling(in_channels, out_channels, kernel_size = 2, stride = 2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True)
        )
class UNet(nn.Module):
    def __init__(self, in_ch, n_classes):
        super.__init__()
        #go down
        self.conv1 = conv_twice(in_ch, 64)
        self.conv2 = conv_twice(64, 128)
        self.conv3 = conv_twice(128, 256)
        self.conv4 = conv_twice(256, 512)
        self.conv5 = conv_twice(512, 1024)
        self.down_pooing = nn.MaxPool2d(2)
        
        #go up
        self.up_pool6 = up_pooling(1024, 512)
        self.conv_up6 = conv_twice(1024, 512)
        self.up_pool7 = up_pooling(512, 256)
        self.conv_up7 = conv_twice(512, 256)
        self.up_pool8 = up_pooling(256, 128)
        self.conv_up8 = conv_twice(256, 128)
        self.up_pool9 = up_pooling(128, 64)
        self.conv_up9 = conv_twice(128, 64)
        
        self.conv_up10 = nn.Conv2d(64, n_classes, 1)
        
    def forward(self, x):
        
        x1 = self.conv1(x)
        p1 = self.down_pooing(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooing(x2)
        x3 = self.conv3(p2)
        p3 = self.downpooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooing(x4)
        x5 = self.conv5(p4)
        
        p6 = self.conv_up6(x5)
        x6 = torch.cat([p6, x4], dim = 1)
        x6 = self.conv_up6(x6)
        
        p7 = self.conv_up7(x6)
        x7 = torch.cat([p7, x3], dim = 1)
        x7 = self.conv_up7(x7)
        
        p8 = self.conv_up8(x7)
        x8 = torch.cat([p8, x2], dim = 1)
        x8 = self.conv_up8(x8)
        
        p9 = self.conv_up9(x8)
        x9 = torch.cat([p9, x1], dim = 1)
        x9 = self.conv_up9(x9)
        
        output = self.conv10(x9)
        output = nn.Sigmoid(output)
        
        return output