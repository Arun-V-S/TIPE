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
from tkinter import *
from PIL import Image, ImageTk
import wandb
import time

wandb.init(project = 'Bateaux')
##Ici, on cherche juste un couple (x, y), pas une bounding box!
#Tf2-bon
Win = Tk()
affichage = Frame(Win, width = 512, height = 512)
imageCanvas = Canvas(affichage, width = 512, height = 512)
imageIndice = StringVar()
nombreTrain = StringVar()
imageIndice.set('1')

buttonFrame = Frame(Win)


ABSOLUTE = 'D:/Documents/Prepa/TIPE'

pathBateaux = ABSOLUTE + "/data/MASATI-v2/ship"
pathXml = ABSOLUTE + "/data/MASATI-v2/ship_labels"
pathMer = ABSOLUTE + "/data/MASATI-v2/water"
pathModels = ABSOLUTE + "/Models/"
pathXml = ABSOLUTE + "/data/MASATI-v2/ship_labels"

listeBateaux = os.listdir(pathBateaux)
listeMer = os.listdir(pathMer)

NUMBER = 250
TOTAL = 300

bateauxCsvPos.generateCsv(NUMBER, TOTAL)
print(".csv généré")

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
                    xmin = float(xmin) / 512
                    xmax = float(xmax) / 512
                    ymin = float(ymin) / 512
                    ymax = float(ymax) / 512
                    self.images.append(self.transform(cv2.imread(image)).float())
                    self.resultats.append(torch.Tensor([(xmin + xmax) / 2, (ymin + ymax) / 2]))

    def __getitem__(self, index):
        """image = self.transform(cv2.imread(self.images[index])).float()"""
        image = self.images[index]
        resultat = self.resultats[index]
        return image, resultat

    def __len__(self):
        return len(self.resultats)

set_images = ImageData("D:/Documents/Prepa/TIPE/bateauxPos.csv", transforms.Compose([transforms.ToTensor(),]))
imagesLoader = torch.utils.data.DataLoader(set_images, batch_size = 4, shuffle = True, pin_memory=True, num_workers=0)
print("Set de train chargé")

set_images_val = ImageData("D:/Documents/Prepa/TIPE/bateauxPosVal.csv", transforms.Compose([transforms.ToTensor(),]))
imagesLoader = torch.utils.data.DataLoader(set_images, batch_size = 8, shuffle = True, pin_memory=True, num_workers=0)
print("Set de validation chargé")

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

        self.conv = nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, 1),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 64, 3, 1, 1),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(64, 128, 3, 1, 1),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(inplace=False),
        nn.Linear(512, 10),
        nn.ReLU(inplace=False),
        nn.Linear(10, 2),
        #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)

        x = self.review(x)

        x = self.classifier(x)
        return x

    def review(self, x):
        return x.view(-1, 2048)




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
        wandb.log({'epoch' : epoch, 'loss' : running_loss})
        net.epochs += 1
        #afficherPreview()
        #imageCanvas.update()


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
                image, res = set_images[i][0].unsqueeze(0).to(device), set_image[i][1]
                xmin = res[0]
                xmax = res[0]
                ymin = res[0]
                ymax = res[0]
                x = (xmin + xmax) / 2
                y = (ymin + ymax) / 2
                result = net(image).detach().cpu()[0]
                res = (abs(x - result[0]) + abs(y - result[1])) / 2
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
                x = (xmin + xmax) / 2
                y = (ymin + ymax) / 2
                image = set_images_val[i][0].unsqueeze(0).to(device)
                result = net(image).detach().cpu()[0]
                res = (abs(x - result[0]) + abs(y - result[1])) / 2
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

net = Net()

net.load_state_dict(torch.load(pathModels + 'Tf1'))


for param in net.parameters():
    param.requires_grad = False

net.classifier = nn.Sequential(
nn.Linear(2048, 512),
nn.ReLU(),
nn.Linear(512, 2),
nn.Sigmoid()
)




params = [p for p in net.parameters()]
net.to(device, non_blocking=True)
criterion = nn.L1Loss()
optimizer = optim.Adam(params, lr=0.001, betas=(0.9, 0.999))
#optimizer = optim.SGD(params, lr = 0.005, momentum = 0.9, weight_decay = 1e-6)

wandb.watch(net, log="all")

##Affichage
def corrigerStr(chaine):
    if len(chaine) >= 4:
        return chaine
    else:
        return corrigerStr('0' + chaine)

def incr():
    global imageIndice
    res = (int(imageIndice.get()) + 1) % TOTAL
    if res == 0:
        res = 1
    imageIndice.set(str(res))
    afficherPreview()

def decr():
    global imageIndice
    res = (int(imageIndice.get()) - 1) % TOTAL
    if res == 0:
        res = 1
    imageIndice.set(str(res))
    afficherPreview()

def afficherPreview():
    global imageIndice
    indiceStr = corrigerStr(imageIndice.get())
    nom = 's' + indiceStr + '.xml'
    tree = etree.parse(pathXml + '/' + nom)
    xmin = tree.xpath('/annotation/object/bndbox/xmin')[0].text
    xmax = tree.xpath('/annotation/object/bndbox/xmax')[0].text
    ymin = tree.xpath('/annotation/object/bndbox/ymin')[0].text
    ymax = tree.xpath('/annotation/object/bndbox/ymax')[0].text
    x0 = (int(xmin) + int(xmax)) // 2
    y0 = (int(ymin) + int(ymax)) // 2

    imageI = Image.open(pathBateaux + '/s' + indiceStr + '.png')
    initI = np.array(imageI)
    init = initI.transpose((2, 0, 1))
    init = torch.tensor(init).float()
    init = init.to(device)
    init = init.unsqueeze(0)
    result = net(init).detach().cpu().numpy()
    x, y = result[0]
    x = int(x * 512)
    y = int(y * 512)
    imageCanvas.delete('all')
    image1 = Image.fromarray(initI)
    photo1 = ImageTk.PhotoImage(image1)
    imageCanvas.create_image(0, 0, anchor = NW, image = photo1)
    imageCanvas.image = photo1
    imageCanvas.create_oval(x0 - 2, y0 - 2, x0 + 2, y0 + 2, outline = 'green', width = 6)
    imageCanvas.create_oval(x - 2, y - 2, x + 2, y + 2, outline = 'red', width = 6)

def gotrain():
    nombre = int(nombreEntry.get())
    train(nombre)

def setTrain():
    imageIndice.set(str(1))
    afficherPreview()

def setVal():
    imageIndice.set(str(TOTAL - NUMBER))
    afficherPreview()

incrButton = Button(buttonFrame, command = incr, text = "Suivant")
decrButton = Button(buttonFrame, command = decr, text = 'Précédent')
nombreEntry = Entry(buttonFrame, textvariable = nombreTrain)
trainButton = Button(buttonFrame, command = gotrain, text = 'Train!')
goToTrainButton = Button(buttonFrame, command = setTrain, text = 'go to train')
goToValButton = Button(buttonFrame, command = setVal, text = 'go to val')

decrButton.pack(side = LEFT)
incrButton.pack(side = LEFT)
nombreEntry.pack(side = LEFT)
trainButton.pack(side = LEFT)
goToTrainButton.pack(side = LEFT)
goToValButton.pack(side = LEFT)

imageCanvas.pack(side = TOP)
affichage.pack(side = TOP)
buttonFrame.pack(side = TOP)

print(count_parameters(net))
#loadModel('Pos2')
#Win.mainloop()

train(100)
