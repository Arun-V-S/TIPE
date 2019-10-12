from __future__ import print_function, division
import os
import cv2
import csv
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
import bateauxCsvPos
from torchstat import stat
from torchvision.utils import make_grid
from lxml import etree



ABSOLUTE = 'D:/Documents/Prepa/TIPE'

pathBateaux = ABSOLUTE + "/data/MASATI-v2/ship"
pathXml = ABSOLUTE + "/data/MASATI-v2/ship_labels"
pathMer = ABSOLUTE + "/data/MASATI-v2/water"
pathModels = ABSOLUTE + "/Models/"

listeBateaux = os.listdir(pathBateaux)
listeMer = os.listdir(pathMer)

NUMBER = 256
TOTAL = 1024

bateauxCsvPos.generateCsv(NUMBER)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ImageData(Dataset):
    def __init__(self, csvtruc, transform = None):
        self.taille = 0
        self.transform = transform
        self.images = []
        self.resultats = []
        with open(csvtruc, 'r') as fichier:
            truc = csv.reader(fichier, delimiter = ',')
            for ligne in truc:
                if ligne != []:
                    self.taille += 1
                    image, xmin, xmax, ymin, ymax = ligne[0].split(',')
                    self.images.append(self.transform(cv2.imread(image)).float())
                    self.resultats.append(torch.Tensor([float(xmin) / 512, float(xmax) / 512, float(ymin) / 512, float(ymax) / 512]))

    def __getitem__(self, index):
        """image = self.transform(cv2.imread(self.images[index])).float()"""
        image = self.images[index]
        resultat = self.resultats[index]
        return image, resultat

    def __len__(self):
        return len(self.resultats)

set_images = ImageData("D:/Documents/Prepa/TIPE/bateauxPos.csv", transforms.Compose([transforms.ToTensor(),]))
imagesLoader = torch.utils.data.DataLoader(set_images, batch_size = 8, shuffle = True, pin_memory=True, num_workers=0)

set_images_val = ImageData("D:/Documents/Prepa/TIPE/bateauxPosVal.csv", transforms.Compose([transforms.ToTensor(),]))
imagesLoader = torch.utils.data.DataLoader(set_images, batch_size = 8, shuffle = True, pin_memory=True, num_workers=0)

def load():
    global set_images
    global imagesLoader
    set_images = ImageData("D:/Documents/Prepa/TIPE/bateauxPos.csv", transforms.Compose([transforms.ToTensor()]))
    imagesLoader = torch.utils.data.DataLoader(set_images, batch_size = 32, shuffle = True, pin_memory=True, num_workers=0)
    print('Images chargées.')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.epochs = 0

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, 3, 1, 1)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu3 = nn.ReLU()
        self.norm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu4 = nn.ReLU()
        self.norm4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu5 = nn.ReLU()
        self.norm5 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv6 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu6 = nn.ReLU()
        self.norm6 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu7 = nn.ReLU()
        self.norm7 = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu8 = nn.ReLU()
        self.norm8 = nn.BatchNorm2d(128)
        self.pool6 = nn.MaxPool2d(2, 2)

        self.conv9 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu9 = nn.ReLU()
        self.norm9 = nn.BatchNorm2d(128)
        self.pool7 = nn.MaxPool2d(2, 2)


        self.lin1 = nn.Linear(2048, 4)
        self.relu10 = nn.ReLU()

    def forward(self, x):
        x = self.norm1(self.relu1(self.conv1(x)))
        x = self.norm2(self.relu2(self.conv2(x)))
        x = self.pool1(x)

        x = self.norm3(self.relu3(self.conv3(x)))
        x = self.norm4(self.relu4(self.conv4(x)))
        x = self.pool2(x)

        x = self.pool3(self.norm5(self.relu5(self.conv5(x))))

        x = self.pool4(self.norm6(self.relu6(self.conv6(x))))

        x = self.pool5(self.norm7(self.relu7(self.conv7(x))))

        x = self.pool6(self.norm8(self.relu8(self.conv8(x))))

        x = self.pool7(self.norm9(self.relu9(self.conv9(x))))


        x = x.view(-1, 2048)
        x = self.relu10(self.lin1(x))
        return x

net = Net()
net.to(device, non_blocking=True)
criterion = nn.MSELoss()
optimizer = optim.SGD(list(net.parameters()), lr = 0.001, momentum = 0.9)

"""
for i in enumerate(imagesLoader):
    M = i

I = M[1][0][0].to(device)
J = net(I.unsqueeze(0))
print(J.shape)
"""



def train(number):
    for epoch in range(number):
        running_loss = 0.0
        for i, data in enumerate(imagesLoader, 0):
            input, expected = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = net(input)
            loss = criterion(outputs, expected)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch : ' + str(epoch) + ' loss : ' + str(running_loss))
        net.epochs += 1


def testPos():
    global NUMBER
    global TOTAL
    global set_image
    prec_Tr = 0
    prec_Val = 0
    with open(ABSOLUTE + '/bateauxPos.csv', 'r') as fichier:
        truc = csv.reader(fichier, delimiter = ',')
        i = 0
        for ligne in truc:
            if ligne != []:
                """_, xmin, xmax, ymin, ymax = ligne[0].split(',')
                xmin = float(xmin) / 512
                xmax = float(xmin) / 512
                ymin = float(xmin) / 512
                ymax = float(xmin) / 512"""
                image, res = set_images[i][0].unsqueeze(0).to(device), set_image[i][1]
                xmin = res[0]
                xmax = res[0]
                ymin = res[0]
                ymax = res[0]
                result = net(image).detach().cpu()[0]
                res = ((abs(xmin - result[0])) + (abs(xmax - result[1])) + (abs(ymin - result[2])) + (abs(ymax - result[3]))) / 4
                prec_Tr += res / NUMBER
                i += 1
    with open(ABSOLUTE + '/bateauxPosVal.csv', 'r') as fichier:
        truc = csv.reader(fichier, delimiter = ',')
        i = 0
        for ligne in truc:
            if ligne != []:
                _, xmin, xmax, ymin, ymax = ligne[0].split(',')
                xmin = float(xmin) / 512
                xmax = float(xmin) / 512
                ymin = float(xmin) / 512
                ymax = float(xmin) / 512
                image = set_images_val[i][0].unsqueeze(0).to(device)
                result = net(image).detach().cpu()[0]
                res = ((abs(xmin - result[0])) + (abs(xmax - result[1])) + (abs(ymin - result[2])) + (abs(ymax - result[3]))) / 4
                prec_Val += res / (TOTAL - NUMBER)
                i += 1
    return prec_Tr, prec_Val



def saveModel(nom):
    torch.save(net.state_dict(), pathModels + nom)

def loadModel(nom):
    net.load_state_dict(torch.load(pathModels + nom))

def show(layer, number, imageN):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    net.conv1.register_forward_hook(get_activation('conv1'))
    net.conv2.register_forward_hook(get_activation('conv2'))
    net.conv3.register_forward_hook(get_activation('conv3'))
    net.conv4.register_forward_hook(get_activation('conv4'))
    net.conv5.register_forward_hook(get_activation('conv5'))
    net.conv6.register_forward_hook(get_activation('conv6'))
    net.conv7.register_forward_hook(get_activation('conv7'))
    net.conv8.register_forward_hook(get_activation('conv8'))
    net.conv9.register_forward_hook(get_activation('conv9'))
    data = set_images[imageN][0].to(device, non_blocking=True)
    output = net(data.unsqueeze(0))

    act = activation['conv' + str(layer)].squeeze().cpu()
    fir, axarr = plt.subplots(number)
    for idx in range(number):
        axarr[idx].imshow(act[idx])
    plt.show()

def show2(number):
    kernels = net.conv1.weight.detach().cpu()
    fig, axarr = plt.subplots(number)
    for idx in range(number):
        axarr[idx].imshow(kernels[idx].squeeze())
    plt.show()

def show3():
    kernels = net.conv2.weight.detach().cpu().clone()
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    print(kernels.shape)
    img = make_grid(kernels)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
