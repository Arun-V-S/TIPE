import os
import numpy as np
from TestPerlin import *
from usuelles import *
from PIL import Image
import cv2
import csv

ABSOLUTE = 'D:/Documents/Prepa/TIPE/Images'

pathNormal = ABSOLUTE + "/Normal/"
pathAltered = ABSOLUTE + "/Altered/"
pathPatch = ABSOLUTE + "/Patch/"
pathImage = ABSOLUTE + '/x128/'
pathEncode = ABSOLUTE + '/autoencoder/images/'

SIZE = 128
NUMBER = 2500
NUMBERPATCHS = 500

## Génération images normales
def printMatGrad(matrice, nom, couleurs = "gradient.png"):
    """Retourne la matrice de couleurs avec le gradient en entrée."""
    mini = np.ndarray.min(matrice)
    maxi = np.ndarray.max(matrice)
    n = len(matrice)
    image = Image.new('RGB', (n, n))
    gradient = Image.open(couleurs)
    lGrad = gradient.size[0]
    for x in range(n):
        for y in range(n):
            image.putpixel((x, y), gradient.getpixel((map(matrice[x][y], (mini, maxi), (0, lGrad - 1)), 1)) )
    image.save(nom + '.png')

def generate():
    global NUMBER
    for i in range(500, NUMBER):
        printMatGrad(generate_octaves((SIZE, SIZE, 1), 2, 0.6, 5), pathEncode + str(i))
        if i % 100 == 0:
            print(i)

## Génération patchs
def printPatch(matrice, nom):
    mini = np.ndarray.min(matrice)
    maxi = np.ndarray.max(matrice)
    n = len(matrice)
    moyenne = (mini * 1.3 + maxi * 0.7) / 2
    image = Image.new('RGB', (n, n))
    for x in range(n):
        for y in range(n):
            if matrice[x][y] < moyenne:
                image.putpixel((x, y), (0, 0, 0))
            else:
                image.putpixel((x, y), (255, 255, 255))
    image.save(nom + '.png')

def generatePatchs():
    global NUMBERPATCHS
    for i in range(NUMBERPATCHS):
        printPatch(generate_octaves((SIZE, SIZE, 1), 2, 0.6, 2), pathPatch + str(i))
        if i % 100 == 0:
            print(i)

##Altération des Images
def normalize(array):
    return array / 255.0
def generateAlterates():
    global NUMBER
    for i in range(NUMBER):
        IM = normalize(np.array(Image.open(pathNormal + str(i) + '.png')))
        patch = normalize(cv2.imread(pathPatch + str(np.random.randint(0, NUMBERPATCHS)) + '.png'))
        IM = np.array(IM * patch * 255, dtype = 'i')
        res = Image.fromarray(IM.astype('uint8'))
        res.save(pathAltered + str(i) + '.png')
        if i % 1000 == 0:
            print(i)

##Génération du fichier csv
def generateCsv():
    with open(ABSOLUTE + 'infos.csv', 'w') as fichier:
        filewriter = csv.writer(fichier, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(NUMBER):
            filewriter.writerow([str(pathNormal) + str(i) + '.png,0'])
        for i in range(NUMBER):
            filewriter.writerow([str(pathAltered) + str(i) + '.png,1'])
