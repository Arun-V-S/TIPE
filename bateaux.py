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
import bateauxCsv
from torchstat import stat
from torchvision.utils import make_grid



ABSOLUTE = 'D:/Documents/Prepa/TIPE'

pathBateaux = ABSOLUTE + "/data/MASATI-v2/ship"
pathMer = ABSOLUTE + "/data/MASATI-v2/water"
pathModels = ABSOLUTE + "/Models/"

listeBateaux = os.listdir(pathBateaux)
listeMer = os.listdir(pathMer)

NUMBER = 500

bateauxCsv.generateCsv(NUMBER)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ImageData(Dataset):
    def __init__(self, csvtruc, transform = None):
        self.transform = transform
        self.images = []
        self.resultats = []
        with open(csvtruc, 'r') as fichier:
            truc = csv.reader(fichier, delimiter = ',')
            for ligne in truc:
                if ligne != []:
                    image, resultat = ligne[0].split(',')
                    self.images.append(self.transform(cv2.imread(image)).float())
                    resultat = int(resultat)
                    if resultat == 0:
                        self.resultats.append(0)
                    else:
                        self.resultats.append(1)

    def __getitem__(self, index):
        image = self.images[index]
        resultat = self.resultats[index]
        return image, resultat

    def __len__(self):
        return len(self.resultats)

set_images = ImageData("D:/Documents/Prepa/TIPE/bateaux.csv", transforms.Compose([transforms.ToTensor(),]))
imagesLoader = torch.utils.data.DataLoader(set_images, batch_size = 8, shuffle = True, pin_memory=True, num_workers=0)

print("Images chargées")

def load():
    global set_images
    global imagesLoader
    set_images = ImageData("D:/Documents/Prepa/TIPE/bateaux.csv", transforms.Compose([transforms.ToTensor()]))
    imagesLoader = torch.utils.data.DataLoader(set_images, batch_size = 32, shuffle = True, pin_memory=True, num_workers=0)
    print('Images chargées.')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.epochs = 0

        self.conv = nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 64, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(64, 128, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
        nn.ReLU(),
        nn.Linear(10, 2),
        )

    def forward(self, x):
        x = self.conv(x)

        x = self.review(x)

        x = self.classifier(x)
        return x

    def review(self, x):
        return x.view(-1, 2048)

net = Net()
net.to(device, non_blocking=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(list(net.parameters()), lr = 0.001, momentum = 0.9)



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

def test(altered, numero):
    if altered:
        image = set_images[NUMBER + numero][0].unsqueeze(0).to(device)
    else:
        image = set_images[numero][0].unsqueeze(0).to(device)
    return net(image)

def testSome(Number):
    totalChaque = Number
    global NUMBER
    bateau = 0
    for i in np.random.randint(0, NUMBER, size = (Number,)):
        res = test(True, i)
        if res[0][1] > res[0][0]:
            bateau += 1
    pas_bateau = 0
    for i in np.random.randint(0, NUMBER, size = (Number,)):
        res = test(False, i)
        if res[0][1] < res[0][0]:
            pas_bateau += 1
    print("Pour normal : " + str(pas_bateau / totalChaque) + " et altéré : " + str(bateau / totalChaque))

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
