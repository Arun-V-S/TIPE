from __future__ import print_function, division
import os
import random
import argparse
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dSet
from IPython.display import HTML
import matplotlib.animation as animation
from torchvision import transforms, utils

seed = 123321
random.seed(seed)
torch.manual_seed(seed)

##Hyperparamètres
ABSOLUTE = 'D:/Documents/Prepa/TIPE'
pathImage = ABSOLUTE + '/Images/x128/'
pathModels = ABSOLUTE + "/Models/"

batchSize = 128
imSize = 128 #Ok
channelsNumber = 3 #Couleurs !
inputSize = 100 #Entrée du générateur
featuresGenerator = 64
featuresDiscriminator = 64
learningRate = 0.0002
beta1 = 0.5


setImages = dSet.ImageFolder(root = pathImage, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]))
imagesLoader = torch.utils.data.DataLoader(setImages, batch_size = batchSize, shuffle = True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def weightsInit(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

## générateur

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(inputSize, featuresGenerator * 16, 4, 1, 0, bias = False),
            nn.BatchNorm2d(featuresGenerator * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(featuresGenerator * 16, featuresGenerator * 8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(featuresGenerator * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(featuresGenerator * 8, featuresGenerator * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(featuresGenerator * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(featuresGenerator * 4, featuresGenerator * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(featuresGenerator * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(featuresGenerator * 2, featuresGenerator, 4, 2, 1, bias = False),
            nn.BatchNorm2d(featuresGenerator),
            nn.ReLU(True),

            nn.ConvTranspose2d(featuresGenerator, channelsNumber, 4, 2, 1, bias = False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channelsNumber, featuresDiscriminator, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(featuresDiscriminator, featuresDiscriminator * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(featuresDiscriminator * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(featuresDiscriminator * 2, featuresDiscriminator * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(featuresDiscriminator * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(featuresDiscriminator * 4, featuresDiscriminator * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(featuresDiscriminator * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(featuresDiscriminator * 8, featuresDiscriminator * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(featuresDiscriminator * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(featuresDiscriminator * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

netG = Generator().to(device)
netG.apply(weightsInit)

netD = Discriminator().to(device)
netD.apply(weightsInit)


criterion = nn.BCELoss()
fixedNoise = torch.randn(128, inputSize, 1, 1, device = device)

realLabel = 1
fakeLabel = 0

optimD = optim.Adam(netD.parameters(), lr = learningRate, betas = (beta1, 0.999))
optimG = optim.Adam(netG.parameters(), lr = learningRate, betas = (beta1, 0.999))


imgList = []
GLoss = []
DLoss = []


def train(number):
    iters = 0
    for epoch in range(number):
        for i, data in enumerate(imagesLoader, 0):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), realLabel, device = device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, inputSize, 1, 1, device = device)
            fake = netG(noise)
            label.fill_(fakeLabel)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimD.step()

            netG.zero_grad()
            label.fill_(realLabel)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimG.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, number, i, len(imagesLoader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            GLoss.append(errG.item())
            DLoss.append(errD.item())
            if (iters % 500 == 0) or ((epoch == number) and (i == len(imagesLoader)-1)):
                with torch.no_grad():
                    fake = netG(fixedNoise).detach().cpu()
                imgList.append(vutils.make_grid(fake, padding=2, normalize=True))
            iters += 1

def show():
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in imgList]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())
    plt.show()

def clear():
    del imagesLoader

def saveModel(nom):
    torch.save(netD.state_dict(), pathModels +  'D-' + nom)
    torch.save(netG.state_dict(), pathModels +  'G-' + nom)

def loadModel(nom):
    netD.load_state_dict(torch.load(pathModels + 'D-' + nom))
    netG.load_state_dict(torch.load(pathModels + 'G-' + nom))
    netD.eval()
    netG.eval()
