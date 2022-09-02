import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.autograd

from tensorboardX import SummaryWriter

class Binarization(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input):
        return torch.round(input)

    @staticmethod
    def backward(ctx,grad_output):
        return grad_output

class MaskGeneratorNet(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        ###### C-Block Stacking ##########
        self.l11 = nn.Conv2d(in_channels, 32, 5, padding='same')
        self.l12 = nn.Conv2d(in_channels, 16, 5, padding='same', dilation=2)
        self.l13 = nn.Conv2d(in_channels, 16, 5, padding='same', dilation=5)
        self.d1 = nn.MaxPool2d(2)
        self.l21 = nn.Conv2d(64, 64, 5, padding='same')
        self.l22 = nn.Conv2d(64, 32, 5, padding='same', dilation=2)
        self.l23 = nn.Conv2d(64, 32, 5, padding='same', dilation=5)
        self.d2 = nn.MaxPool2d(2)
        self.l31 = nn.Conv2d(128, 128, 5, padding='same')
        self.l32 = nn.Conv2d(128, 64, 5, padding='same', dilation=2)
        self.l33 = nn.Conv2d(128, 64, 5, padding='same', dilation=5)
        self.d3 = nn.MaxPool2d(2)
        
        
        self.l41 = nn.Conv2d(256, 256, 5, padding='same')
        self.l42 = nn.Conv2d(256, 128, 5, padding='same', dilation=2)
        self.l43 = nn.Conv2d(256, 128, 5, padding='same', dilation=5)
        
        
        self.u1 = nn.Upsample(scale_factor=2)
        self.tl11 = nn.ConvTranspose2d(256+512, 128, 5, padding=2)
        self.tl12 = nn.ConvTranspose2d(256+512, 64, 5, dilation=2, padding=4)
        self.tl13 = nn.ConvTranspose2d(256+512, 64, 5, dilation=5, padding=10)
        self.u2 = nn.Upsample(scale_factor=2)
        self.tl21 = nn.ConvTranspose2d(128+256, 64, 5, padding=2)
        self.tl22 = nn.ConvTranspose2d(128+256, 32, 5, padding=4, dilation=2)
        self.tl23 = nn.ConvTranspose2d(128+256, 32, 5, padding=10, dilation=5)
        self.u3 = nn.Upsample(scale_factor=2)
        self.tl31 = nn.ConvTranspose2d(64+128, 32, 5, padding=2)
        self.tl32 = nn.ConvTranspose2d(64+128, 16, 5, padding=4, dilation=2)
        self.tl33 = nn.ConvTranspose2d(64+128, 16, 5, padding=10, dilation=5)
        ###### Conv Layer Stacking ##########
        self.conv1 = nn.ConvTranspose2d(64+in_channels,8, 3, padding=1)
        self.conv2 = nn.ConvTranspose2d(8, 1, 3, padding=1)
        self.mask = Binarization.apply

    def forward(self, x):
        #### Conv Block 1
        temp1 = F.elu(self.l11(x))
        temp2 = F.elu(self.l12(x))
        temp3 = F.elu(self.l13(x))
        x1o = torch.cat([temp1,temp2, temp3], dim=1)
        out = self.d1(x1o)
        
        #### Conv Block 2
        temp1 = F.elu(self.l21(out))
        temp2 = F.elu(self.l22(out))
        temp3 = F.elu(self.l23(out))
        x2o = torch.cat([temp1,temp2, temp3], dim=1)
        out = self.d2(x2o)
        
        #### Conv Block 3
        temp1 = F.elu(self.l31(out))
        temp2 = F.elu(self.l32(out))
        temp3 = F.elu(self.l33(out))
        x3o = torch.cat([temp1,temp2, temp3], dim=1)
        out = self.d3(x3o)

        #### Conv Block 4
        temp1 = F.elu(self.l41(out))
        temp2 = F.elu(self.l42(out))
        temp3 = F.elu(self.l43(out))
        x4o = torch.cat([temp1,temp2, temp3], dim=1)
        out = self.u1(x4o) 
        
        #### Tr-Conv Block 1
        out = torch.cat([out, x3o], dim=1)
        temp1 = F.elu(self.tl11(out))
        temp2 = F.elu(self.tl12(out))
        temp3 = F.elu(self.tl13(out))
        out = torch.cat([temp1, temp2, temp3], dim=1)
        out = self.u2(out)
        
        #### Tr-Conv Block 2
        out = torch.cat([out, x2o], dim=1)
        temp1 = F.elu(self.tl21(out))
        temp2 = F.elu(self.tl22(out))
        temp3 = F.elu(self.tl23(out))
        out = torch.cat([temp1, temp2, temp3], dim=1)
        out = self.u3(out)
        
        #### Tr-Conv Block 3
        out = torch.cat([out, x1o], dim=1)
        temp1 = F.elu(self.tl31(out))
        temp2 = F.elu(self.tl32(out))
        temp3 = F.elu(self.tl33(out))
        out = torch.cat([temp1, temp2, temp3, x], dim=1)
        
        #### Conv Layer Head 1
        out = F.elu(self.conv1(out))
        
        out = F.hardsigmoid(self.conv2(out))
        
        out = self.mask(out)
        return out

    
class InpaintingNet(nn.Module):
    def __init__(self, in_channels=7):
        super().__init__()
        ###### C-Block Stacking ##########
        self.l11 = nn.Conv2d(in_channels, 32, 5, padding='same')
        self.l12 = nn.Conv2d(in_channels, 16, 5, padding='same', dilation=2)
        self.l13 = nn.Conv2d(in_channels, 16, 5, padding='same', dilation=5)
        self.d1 = nn.MaxPool2d(2)
        self.l21 = nn.Conv2d(64, 64, 5, padding='same')
        self.l22 = nn.Conv2d(64, 32, 5, padding='same', dilation=2)
        self.l23 = nn.Conv2d(64, 32, 5, padding='same', dilation=5)
        self.d2 = nn.MaxPool2d(2)
        self.l31 = nn.Conv2d(128, 128, 5, padding='same')
        self.l32 = nn.Conv2d(128, 64, 5, padding='same', dilation=2)
        self.l33 = nn.Conv2d(128, 64, 5, padding='same', dilation=5)
        self.d3 = nn.MaxPool2d(2)
        
        
        self.l41 = nn.Conv2d(256, 256, 5, padding='same')
        self.l42 = nn.Conv2d(256, 128, 5, padding='same', dilation=2)
        self.l43 = nn.Conv2d(256, 128, 5, padding='same', dilation=5)
        
        
        self.u1 = nn.Upsample(scale_factor=2)
        self.tl11 = nn.ConvTranspose2d(256+512, 128, 5, padding=2)
        self.tl12 = nn.ConvTranspose2d(256+512, 64, 5, dilation=2, padding=4)
        self.tl13 = nn.ConvTranspose2d(256+512, 64, 5, dilation=5, padding=10)
        self.u2 = nn.Upsample(scale_factor=2)
        self.tl21 = nn.ConvTranspose2d(128+256, 64, 5, padding=2)
        self.tl22 = nn.ConvTranspose2d(128+256, 32, 5, padding=4, dilation=2)
        self.tl23 = nn.ConvTranspose2d(128+256, 32, 5, padding=10, dilation=5)
        self.u3 = nn.Upsample(scale_factor=2)
        self.tl31 = nn.ConvTranspose2d(64+128, 32, 5, padding=2)
        self.tl32 = nn.ConvTranspose2d(64+128, 16, 5, padding=4, dilation=2)
        self.tl33 = nn.ConvTranspose2d(64+128, 16, 5, padding=10, dilation=5)
        ###### Conv Layer Stacking ##########
        self.conv1 = nn.ConvTranspose2d(64+in_channels,8, 3, padding=1)
        self.conv2 = nn.ConvTranspose2d(8, 3, 3, padding=1)

    def forward(self, x):
        #### Conv Block 1
        temp1 = F.elu(self.l11(x))
        temp2 = F.elu(self.l12(x))
        temp3 = F.elu(self.l13(x))
        x1o = torch.cat([temp1,temp2, temp3], dim=1)
        out = self.d1(x1o)
        
        #### Conv Block 2
        temp1 = F.elu(self.l21(out))
        temp2 = F.elu(self.l22(out))
        temp3 = F.elu(self.l23(out))
        x2o = torch.cat([temp1,temp2, temp3], dim=1)
        out = self.d2(x2o)
        
        #### Conv Block 3
        temp1 = F.elu(self.l31(out))
        temp2 = F.elu(self.l32(out))
        temp3 = F.elu(self.l33(out))
        x3o = torch.cat([temp1,temp2, temp3], dim=1)
        out = self.d3(x3o)

        #### Conv Block 4
        temp1 = F.elu(self.l41(out))
        temp2 = F.elu(self.l42(out))
        temp3 = F.elu(self.l43(out))
        x4o = torch.cat([temp1,temp2, temp3], dim=1)
        out = self.u1(x4o) 
        
        #### Tr-Conv Block 1
        out = torch.cat([out, x3o], dim=1)
        temp1 = F.elu(self.tl11(out))
        temp2 = F.elu(self.tl12(out))
        temp3 = F.elu(self.tl13(out))
        out = torch.cat([temp1, temp2, temp3], dim=1)
        out = self.u2(out)
        
        #### Tr-Conv Block 2
        out = torch.cat([out, x2o], dim=1)
        temp1 = F.elu(self.tl21(out))
        temp2 = F.elu(self.tl22(out))
        temp3 = F.elu(self.tl23(out))
        out = torch.cat([temp1, temp2, temp3], dim=1)
        out = self.u3(out)
        
        #### Tr-Conv Block 3
        out = torch.cat([out, x1o], dim=1)
        temp1 = F.elu(self.tl31(out))
        temp2 = F.elu(self.tl32(out))
        temp3 = F.elu(self.tl33(out))
        out = torch.cat([temp1, temp2, temp3, x], dim=1)
        
        #### Conv Layer Head 1
        out = F.elu(self.conv1(out))
        
        out = F.hardsigmoid(self.conv2(out))
        return out


