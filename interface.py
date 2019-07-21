import numpy as np
from tkinter import *
from tkinter.messagebox import *
import cv2
from PIL import Image, ImageTk
import PIL
from math import *
from copy import *

##Fenêtre
WIDTH = 400
HEIGHT = 400
SLIDERLENGTH = 100

##Sliders Hyperparamètres
COLUMNS = 5
CARACS = 20 #Nombre de paramètres ajustables
ROWS = ceil(CARACS / COLUMNS)
#VARIABLES = np.random.rand(CARACS)
myFrame = Tk()

VARIABLES = []
for _ in range(CARACS):
    a = DoubleVar()
    VARIABLES.append(a)

ABSOLUTE = 'D:/Documents/Prepa/TIPE'

pathNormal = ABSOLUTE + "/Images/Normal/"
pathAltered = ABSOLUTE + "/Images/Altered/"
pathModels = ABSOLUTE + "/Models/"



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
    arguments = np.zeros(CARACS)
    for i in range(CARACS):
        arguments[i] = VARIABLES[i].get()
    print(arguments)

imageCanvas = Canvas(myFrame, width = WIDTH, height = HEIGHT)
temp = ImageTk.PhotoImage(Image.open(pathNormal  + '1.png').resize((WIDTH, HEIGHT), PIL.Image.ANTIALIAS))


imageCanvas.create_image(0, 0, anchor = NW, image = temp)
imageCanvas.pack(side = LEFT)

slidersFrameMaster = Frame(myFrame)
#VARIABLES = [1, 2, 3, 4]
##Construction des sliders
SLIDERS = []
for i in range(ROWS):
    sliderFrame = Frame(slidersFrameMaster)
    for j in range(COLUMNS):
        SLIDERS.append(Scale(sliderFrame, from_ = - 1, to = 1, orient = VERTICAL, length = SLIDERLENGTH, resolution = 0.1, tickinterval = 0.1, width = 20, variable = VARIABLES[i * COLUMNS + j]).pack(side = LEFT))
    sliderFrame.pack()

def lol():
    for V in VARIABLES:
        print(V.get())

Bouton = Button(slidersFrameMaster, text = 'lol', command = lol).pack()
slidersFrameMaster.pack(side = RIGHT, padx = 10, pady = 10)

generateButton = Button(slidersFrameMaster, text = 'Générer', command = generer).pack()

myFrame.mainloop()
