import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init

from Module.Normalization import  SpectralNorm

class Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.pre_conv = nn.Sequential(SpectralNorm(nn.Conv2d(in_channels,64,3, padding='same')),
                                       nn.LeakyReLU(),
                                      nn.MaxPool2d(2),
                                       SpectralNorm(nn.Conv2d(64,128,3, padding='same')), 
                                       nn.LeakyReLU(),
                                      nn.MaxPool2d(2),
                                      SpectralNorm(nn.Conv2d(128,256,3, padding='same')),
                                       nn.LeakyReLU(),
                                      nn.MaxPool2d(2),
                                      SpectralNorm(nn.Conv2d(256,512,3, padding='same')),
                                       nn.LeakyReLU(),
                                      nn.MaxPool2d(2)
                                      )
        self.linear = SpectralNorm(nn.Linear(512*8*8, 1))
    def forward(self,x):
        out = self.pre_conv(x)
        B,C,H,W = out.shape
        out = out.view(B,-1)
        out = self.linear(out)
        return out