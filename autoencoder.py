import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.datasets as dSet
import numpy as np
import os
from tkinter import *
from tkinter.messagebox import *
import cv2
from PIL import Image, ImageTk
import PIL
from math import *
from copy import *

ABSOLUTE = 'D:/Documents/Prepa/TIPE/Images'

pathNormal = ABSOLUTE + "/Normal/"
pathAltered = ABSOLUTE + "/Altered/"
pathPatch = ABSOLUTE + "/Patch/"
pathImage = ABSOLUTE + '/x128/'
pathEncode = ABSOLUTE + '/autoencoder/'

batchSize = 128

setImages = dSet.ImageFolder(root = pathEncode, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]))
imagesLoader = torch.utils.data.DataLoader(setImages, batch_size = batchSize, shuffle = True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_img(x):
    x = 0.5 * (x + 1)
    x.squeeze_(0)
    x = x.clamp(0, 1)
    x = x.numpy()
    x = x.transpose((1, 2, 0))
    x = x * 255
    x = x.astype(np.uint8)
    return x

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2
            nn.Conv2d(8, 3, 3, stride = 3, padding = 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride = 1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 8, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 4, stride=3, padding=4),  # b, 1, 88, 88
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def test(self, x):
        x = self.decoder(x)
        return x
    def encode(self, x):
        x = self.encoder(x)
        return x


model = autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

im = np.array(Image.open(pathNormal + '1.png'))
im = im.transpose((2, 0, 1))
im = torch.tensor(im).float().to(device)
im = im.unsqueeze(0)
x = model(im)
print(im.size())
print(model.encode(im).size())
print(x.size())

def train(number):
    for i in range(number):
        for data in imagesLoader:
            img, _ = data
            img = img.to(device)
            output = model(img)
            loss = criterion(output, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch : ' + str(i) + ' loss : ' + str(loss.item()))


##INTERFACE
WIDTH = 800
HEIGHT = 800
SLIDERLENGTH = 100

##Sliders Hyperparamètres
COLUMNS = 8
CARACS = 27 #Nombre de paramètres ajustables
ROWS = ceil(CARACS / COLUMNS)
#VARIABLES = np.random.rand(CARACS)
myFrame = Tk()

VARIABLES = []
for _ in range(CARACS):
    a = DoubleVar()
    VARIABLES.append(a)

myFrame.title('Auto-encoder')

issou = 0

photo = []

def loadNN():
    temp = np.random.randint(0, 255, (WIDTH, HEIGHT), dtype = 'i3').astype(np.uint8)
    return ImageTk.PhotoImage(image = Image.fromarray(temp))

def rechargerImage():
    global imageCanvas
    imageCanvas.delete(ALL)
    imageCanvas.create_image(0, 0, anchor = NW, image = loadNN())

def generer():
    global VARIABLES
    global imageCanvas
    arguments = np.zeros(CARACS)
    for i in range(CARACS):
        arguments[i] = VARIABLES[i].get()
    input = arrayToTensor(arguments)
    output = model.test(input)
    output = to_img(output.cpu().data)
    global image
    image = Image.fromarray(output).resize((WIDTH, HEIGHT), PIL.Image.ANTIALIAS)
    img = ImageTk.PhotoImage(image)
    imageCanvas.create_image(0, 0, anchor = NW, image = img)
    imageCanvas.image = img
    """.resize((WIDTH, HEIGHT), PIL.Image.ANTIALIAS))"""

def train100():
    train(100)

def train1000():
    train(1000)

def tensorToArray(x):
    x = x.cpu().detach().numpy()

def arrayToTensor(x):
    x = torch.tensor(x).float()
    x = x.view(3, 3, 3)
    x.unsqueeze_(0)
    x = x.to(device)
    return x
global imageCanvas
imageCanvas = Canvas(myFrame, width = WIDTH, height = HEIGHT)
imageCanvas.pack(side = LEFT)


slidersFrameMaster = Frame(myFrame)
#VARIABLES = [1, 2, 3, 4]
##Construction des sliders
SLIDERS = []
for i in range(ROWS):
    sliderFrame = Frame(slidersFrameMaster)
    for j in range(COLUMNS):
        if i * COLUMNS + j < CARACS :
            SLIDERS.append(Scale(sliderFrame, from_ = - 100, to = 100, orient = VERTICAL, length = SLIDERLENGTH, resolution = 0.1, tickinterval = 0.1, width = 20, variable = VARIABLES[i * COLUMNS + j]).pack(side = LEFT))
    sliderFrame.pack()

def lol():
    for V in VARIABLES:
        print(V.get())

#Boutons
generateButton = Button(slidersFrameMaster, text = 'Générer', command = generer).pack(side = LEFT)

trainButton100 = Button(slidersFrameMaster, text = 'Train 100 epochs', command = train100).pack(side = LEFT)
trainButton1000 = Button(slidersFrameMaster, text = 'Train 1000 epochs', command = train1000).pack(side = LEFT)
slidersFrameMaster.pack(side = RIGHT, padx = 10, pady = 10)

myFrame.mainloop()
