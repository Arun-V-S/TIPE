import os
from opensimplex import OpenSimplex
from random import *
from PIL import Image
from math import *
import numpy as np
from usuelles import * #Importe les fonctions usuelles
from Perlin3dNumpy import *

COLOURSold = [(43, 85, 255), (153, 101, 45)] #Anciennes couleurs
COLOURS = [(43, 85, 255), (160, 144, 119), (113, 169, 84), (103, 148, 89), (255, 255, 255)]
OCTAVES = 10
FACTEUR = 0.7
SOMME = sum(FACTEUR ** i for i in range(OCTAVES))
DECX = int(random() * (10 ** 5))
DECY = int(random() * (10 ** 5))
WIDTH = 1500
HEIGHT = 1500
simplex = OpenSimplex(seed = int(random() * (10 ** 15)))

def F(i, j):
    """Fonction noise pour utilisation vectorielle."""
    global HEIGHT
    global WIDTH
    global OCTAVES
    global SOMME
    global FACTEUR
    global DECY
    global DECX
    nx = i/WIDTH - 0.5 #Positions normalisées
    ny = j/HEIGHT - 0.5 #entre -0.5 et +0.5
    e = 0
    for k in range(OCTAVES):
        e += abs((FACTEUR ** k) * simplex.noise2d((2 ** k) * nx + DECX, (2 ** k) * ny + DECY))
    return (e / SOMME)

FVect = np.vectorize(F)
noiseVect = np.vectorize(simplex.noise2d)

def GeneVect(taille):
    printMatGrad(np.fromfunction(FVect, (taille, taille)))
    return

def IssouVect(octaves, facteur, taille):
    M = np.fromfunction(lambda i, j : noiseVect(i, j), (taille, taille))
    for k in range(1, octaves):
        M += np.fromfunction(lambda i, j : (facteur ** k) * noiseVect(i * (2 ** k), j * (2 ** k)), (taille, taille))
    return M

def IssouVect2(taille):
    M = generate_fractal_noise_3d((128, taille, taille), (1, 1, 1), 8, 0.7)
    printMatGrad(M[0])
    return True






def Issou(height, width, coeffs, expo):
    """Crée l'image height*width avec coeffs comme coefficients des harmoniques et expo l'exposant correcteur."""
    matrice = np.empty((width, height))
    n = len(coeffs)
    simplex = OpenSimplex(seed = int(random() * (10 ** 15)))
    mat = []
    sum = 0
    decX = int(random() * (10 ** 5))
    decY = int(random() * (10 ** 5))
    for i in range(n):
        sum += coeffs[i] #Calcule la somem des coeffs

    for y in range(0, height):
        for x in range(0, width):
            nx = x/width - 0.5 #Positions normalisées
            ny = y/width - 0.5 #entre -0.5 et +0.5
            e = 0
            for i in range(n):
                e += abs(coeffs[i] * simplex.noise2d((2 ** i) * nx + decX, (2 ** i) * ny + decY))
            e = e / sum #Renormalise avec la somme des coeffs
            e = pow(e, expo) #Sert à plus ou moins accentuer les variations
            matrice[x, y] = e
        #os.system('cls')
        print("Bruit : " + str(int((y / width) * 100)) + "%")
    return matrice



def printMatGrad(matrice, couleurs = "gradient.png"):
    """Retourne la matrice de couleurs avec le gradient en entrée."""
    minimum = np.min(matrice)
    maximum = np.max(matrice)
    tailleY, tailleX = matrice.shape
    gradient = Image.open(couleurs)
    gradientMat = np.array(gradient)
    resultatMat = np.empty(matrice.shape, dtype=(float, 3))
    lGrad = gradient.size[0]
    for x in range(tailleX):
        for y in range(tailleY):
            resultatMat[x, y] = gradientMat[1, map(matrice[x][y], (minimum, maximum), (0, lGrad - 1), True)]
    res = Image.fromarray(resultatMat.astype('uint8'))
    res.save('Issouuu.png')
    return True

def producIteration(n, taille):
    """Génère n images avec des paramètres prédéfinis."""
    for _ in range(n):
        hmin1 = map(random(), (0, 1), (0, 128), True)
        hmax1 = map(random(), (0, 1), (128, 256), True)
        seuil1 = map(random(), (0, 1), (hmin1, hmax1), True)
        #exprintMat(Issou(2000, 2000, (1, 0.75, 0.50, 0.5, 0.25), 3), COLOURS, 128, 0, 256)
        printMatGrad(Issou(taille, taille, (1, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.075, 0.05, 0.025), 2))
        print(str(hmin1) + ' ' + str(seuil1) + ' ' + str(hmax1))
    return
