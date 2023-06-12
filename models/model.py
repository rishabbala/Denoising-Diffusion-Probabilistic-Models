import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
import torch.nn as nn


skip_info = {}

def time_encoding2d(dim, h, w, t):
    assert dim%2 == 0
    pos_emb = torch.zeros(dim, h, w)
    h_dim = int(dim/2)
    pos_emb[0::2, :, :] = torch.sin(t.view(1, -1, 1).repeat(h_dim, h, w))/torch.pow(10000, torch.arange(0, h_dim)*2/dim).view(-1, 1, 1).repeat(1, h, w)
    pos_emb[1::2, :, :] = torch.cos(t.view(1, -1, 1).repeat(h_dim, h, w))/torch.pow(10000, torch.arange(0, h_dim)*2/dim).view(-1, 1, 1).repeat(1, h, w)
    
    return pos_emb


class Sequential(nn.Sequential):
    def forward(self, *args):
        for module in self._modules.values():
            if type(args) == tuple:
                args = module(*args)
            else:
                args = module(args)
        return args


class PixCNNPP(nn.Module):
    
    def __init__(self, num_blocks=2, num_res=4, channels=[4, 16, 32, 64, 128, 256, 512], sz=32):
        super().__init__()                
        self.model = []

        self.upscale = Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=1, stride=1),
            # nn.Dropout2d(p=0.1),
            nn.GroupNorm(num_groups=4, num_channels=4),
            nn.ReLU(),
        )
        
        self.downscale = Sequential(
            nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1, stride=1),
            # nn.Dropout2d(p=0.1),
            # nn.GroupNorm(num_groups=3, num_channels=3),
            # nn.ReLU(),
        )
        
        ## downsample
        for res in range(num_res):
            for block in range(num_blocks):
                if block == 0:
                    self.model.append(DownsampleBlock(in_channels=channels[res], out_channels=channels[res+1], idx=None))
                elif block == num_blocks-1:
                    self.model.append(DownsampleBlock(in_channels=channels[res+1], out_channels=channels[res+1], idx=res))
                else:
                    self.model.append(DownsampleBlock(in_channels=channels[res+1], out_channels=channels[res+1], idx=None))

            self.model.append(ReduceBlock(in_channels=channels[res+1], out_channels=channels[res+1]))
            sz /= 2

        ## Bottleneck
        self.model.append(
            Sequential(
                DownsampleBlock(in_channels=channels[num_res], out_channels=channels[num_res], idx=None),
                DownsampleBlock(in_channels=channels[num_res], out_channels=channels[num_res], idx=None)
            )
        )

        ## upsample
        for res in reversed(range(1, num_res+1)):
            self.model.append(IncreaseBlock(in_channels=channels[res], out_channels=channels[res]))
            sz *= 2
            for block in range(num_blocks):
                if block == 0:
                    self.model.append(UpsampleBlock(in_channels=channels[res], out_channels=channels[res-1], idx=res-1))
                else:
                    self.model.append(UpsampleBlock(in_channels=channels[res-1], out_channels=channels[res-1], idx=None))

        self.model = Sequential(*self.model)
        
        
    def forward(self, x, t):
        global skip_info
        skip_info = {}

        x = self.upscale(x)
        x, _ = self.model(x, torch.tensor([t], requires_grad=False))
        x = self.downscale(x)

        return x
        

class ReduceBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            # nn.Dropout2d(p=0.1),
            nn.GroupNorm(num_groups=out_channels, num_channels=out_channels),
            nn.ReLU()
        )

    
    def forward(self, x, t):
        return self.block(x), t


class IncreaseBlock(nn.Module):

    def __init__(self, in_channels, out_channels, idx=None):
        super().__init__()

        self.block = Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.Dropout2d(p=0.1),
            nn.GroupNorm(num_groups=out_channels, num_channels=out_channels),
            nn.ReLU()
        )

    
    def forward(self, x, t):
        z = self.block(x)
        return z, t


class DownsampleBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, idx=None):
        super().__init__()   
        
        self.idx = idx
        global skip_info
        
        self.block = Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            # nn.Dropout2d(p=0.1),
            nn.GroupNorm(num_groups=out_channels, num_channels=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            # nn.Dropout2d(p=0.1),
            nn.GroupNorm(num_groups=out_channels, num_channels=out_channels)
        )
        self.relu = nn.ReLU()
        
        self.downsample=False
        if in_channels != out_channels:
            self.downsample=True
            self.downsample_block = Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                # nn.Dropout2d(p=0.1),
                nn.GroupNorm(num_groups=out_channels, num_channels=out_channels)
            )

        
    def forward(self, x, t):
        B, C, H, W = x.shape
        pos_emb = time_encoding2d(C, H, W, t)
        x = x + pos_emb.to(x.get_device())
        
        if self.downsample:
            z = self.relu(self.downsample_block(x) + self.block(x))
        else:
            z = self.relu(x + self.block(x))
            
        if self.idx is not None:
            skip_info[self.idx] = z
                
        return z, t
    
    
class UpsampleBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, idx=None):
        super().__init__()    
        
        self.idx = idx
        in_ch = in_channels
        if self.idx is not None:
            in_ch = 2*in_channels
                                      
        self.block = Sequential(
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_channels, kernel_size=3, stride=1, padding=1, output_padding=0),
            # nn.Dropout2d(p=0.1),
            nn.GroupNorm(num_groups=out_channels, num_channels=out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, output_padding=0),
            # nn.Dropout2d(p=0.1),
            nn.GroupNorm(num_groups=out_channels, num_channels=out_channels)
        )
        self.relu = nn.ReLU()
        
        self.upsample=False
        if in_channels != out_channels:
            self.upsample=True
            self.upsample_block = Sequential(
                nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_channels, kernel_size=3, stride=1, padding=1, output_padding=0),
                # nn.Dropout2d(p=0.1),
                nn.GroupNorm(num_groups=out_channels, num_channels=out_channels)
            )
        
        
    def forward(self, x, t):
        global skip_info
        
        B, C, H, W = x.shape
        pos_emb = time_encoding2d(C, H, W, t)
        x = x + pos_emb.to(x.get_device())
        
        if self.idx is not None:
            x = torch.cat((x, skip_info[self.idx]), dim=1)
        
        if self.upsample:
            z = self.relu(self.upsample_block(x) + self.block(x))
        else:
            z = self.relu(x + self.block(x))
        
        return z, t