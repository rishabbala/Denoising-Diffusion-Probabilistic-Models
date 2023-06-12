import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from models.model import PixCNNPP
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from torch.utils.tensorboard import SummaryWriter

def ddp_setup():
    init_process_group(backend='nccl')

def train():
    device = int(os.environ["LOCAL_RANK"])

    if device == 0:
        writer = SummaryWriter(log_dir='./runs')

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=False),
        transforms.RandomHorizontalFlip()
    #     transforms.CenterCrop(224)
    ])

    train_dataset = datasets.Cifar10(root='./datasets', train=True, transform=train_transform, download=False)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    val_dataset = datasets.Cifar10(root='./datasets', train=False, transform=train_transform, download=False)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)

    maxT = 1000
    max_epoch = 10000
    beta_t = torch.arange(0, 2000, 2000/maxT)/(1e5)
    alpha_bar_t = torch.ones(beta_t.shape)
    for i in range(beta_t.shape[0]):
        if i == 0:
            alpha_bar_t[i] = 1-beta_t[i]
        else:
            alpha_bar_t[i] = alpha_bar_t[i-1] * (1-beta_t[i])

    beta_t = beta_t.to(device)
    alpha_bar_t = alpha_bar_t.to(device)

    model = PixCNNPP().to(device)
    L2_loss = torch.nn.MSELoss()
    optim = Adam(model.parameters(), lr=2e-4)
    scheduler = CosineAnnealingLR(optim, max_epoch, eta_min=0, last_epoch=-1, verbose=False)

    print("Training")

    for epoch in range(max_epoch):
        train_loss = 0
        val_loss = 0
        for idx, (img, _) in enumerate(train_dataloader):
            img = img.to(device)
            img = 2*img-1
            t = np.random.choice(np.arange(1, maxT))

            ## forward process
            epsilon = torch.normal(torch.zeros(img.shape), torch.ones(img.shape)).to(device)
            img_t = torch.sqrt(alpha_bar_t[t])*img + torch.sqrt(1-alpha_bar_t[t]) * epsilon

            ## reverse estimation
            epsilon_theta = model(img_t, t)
            loss = L2_loss(epsilon_theta, epsilon)
            train_loss += loss.item()/(len(train_dataloader))

            optim.zero_grad()
            loss.backward()
            optim.step()   

        scheduler.step()
        print(f"Training Loss: {train_loss}")

        # if epoch%100 == 0:
        for idx, (img, _) in enumerate(val_dataloader):
            img = img.to(device)
            img = 2*img-1
            t = np.random.choice(np.arange(1, maxT))

            ## forward process
            epsilon = torch.normal(torch.zeros(img.shape), torch.ones(img.shape)).to(device)
            img_t = torch.sqrt(alpha_bar_t[t])*img + torch.sqrt(1-alpha_bar_t[t]) * epsilon

            ## reverse estimation
            epsilon_theta = model(img_t, t)    
            loss = L2_loss(epsilon_theta, epsilon)
            val_loss += loss.item()/(len(val_dataloader))

        print(f"Val Loss: {val_loss}")
        if device == 0:
            torch.save(model.state_dict(), f'./weights_celeb/model_{epoch}.pth')
            writer.add_scalar('Train Loss', train_loss, epoch)
            writer.add_scalar('Val Loss', val_loss, epoch)
            writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)


if __name__ == '__main__':
    ddp_setup()
    train()
    destroy_process_group()