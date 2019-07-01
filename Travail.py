import os
from opensimplex import OpenSimplex
from random import *
from PIL import Image
from math import *
import numpy as np
from usuelles import * #Importe les fonctions usuelles

COLOURSold = [(43, 85, 255), (153, 101, 45)] #Anciennes couleurs
COLOURS = [(43, 85, 255), (160, 144, 119), (113, 169, 84), (103, 148, 89), (255, 255, 255)]

def ranMat(size, inter, exp, min, max, step):
    """Renvoie la matrice aléatorie de taille size faite avec des fonctions périodiques à coeffs aléatoires, de fréquence min à max avec un pas step."""
    a = 0 #a et b correspondent à la plage des coeffs aléatoires
    b = 5 #des composantes périodiques
    mat = [[]]
    coeffs = []
    nbStep = (max - min) // step
    for i in range(size):
        mat.append([])
        for _ in range(size):
            mat[i].append(1)
    for _ in range(nbStep):
        coeffs.append(map(random(), (0, 1), (a, b)))
    for i in range(nbStep):
        for j in range(size):
            a = 2

def Issou(height, width, coeffs, expo):
    """Crée l'image height*width avec coeffs comme coefficients des harmoniques et expo l'exposant correcteur."""
    n = len(coeffs)
    simplex = OpenSimplex(seed = int(random() * (10 ** 15)))
    mat = []
    min = 100
    max = -1
    sum = 0
    decX = int(random() * (10 ** 5))
    decY = int(random() * (10 ** 5))
    for i in range(n):
        sum += coeffs[i] #Calcule la somem des coeffs

    for y in range(0, height):
        mat.append([])
        for x in range(0, width):
            nx = x/width - 0.5 #Positions normalisées
            ny = y/width - 0.5 #entre -0.5 et +0.5
            e = 0
            for i in range(n):
                e += abs(coeffs[i] * simplex.noise2d((2 ** i) * nx + decX, (2 ** i) * ny + decY))
            e = e / sum #Renormalise avec la somme des coeffs
            e = pow(e, expo) #Sert à plus ou moins accentuer les variations
            mat[y].append(e)
            if e < min:
                min = e
            elif e > max:
                max = e
        #os.system('cls')
        print("Bruit : " + str(int((y / width) * 100)) + "%")
    return (mat, min, max)

def printMat(mat, couleurs, seuil, hmin=0, hmax=256, grad=0, gradDepth=0):
    """Affiche la carte associée à la matrice."""
    (matrice, min, max) = mat
    n = len(matrice)
    image = Image.new('RGB', (n, n))
    for x in range(n):
        for y in range(n):
            valeur = map(matrice[x][y], (min, max), (hmin, hmax), True)
            if valeur < seuil:
                coul = couleurs[0] #Mer
            else:
                coul = couleurs[1] #Terre
            image.putpixel((x, y), coul)
    image.save(str(hmin) + 'x' + str(seuil) + 'x' + str(hmax) + 'x' + str(int(random() * (10 ** 3))) + '.png')
    return True

def exprintMat(mat, couleurs, seuil, hmin=0, hmax=256, grad=0, gradDepth=0):
    """Affiche la carte associée à la matrice avec gestion des biomes."""
    bruit = 5
    MER = couleurs[0]
    TERRE = couleurs[1::] #Couleurs des biomes
    nBiomes = len(TERRE) #Nombre de biomes
    (matrice, min, max) = mat
    n = len(matrice)
    image = Image.new('RGB', (n, n))
    hmaxLoc = hmax - seuil
    for x in range(n):
        for y in range(n):
            valeur = map(matrice[x][y], (min, max), (hmin, hmax), True)
            if valeur < seuil: #Mer
                coul = salt(MER, bruit)
            else:
                valeur = valeur - seuil
                i = 0
                while valeur > i * (hmaxLoc / nBiomes):
                    i += 1
                #numeroCouleur = map(valeur, (hmin, hmax), (1, nBiomes), True)
                coul = salt(couleurs[i], bruit)
            image.putpixel((x, y), coul)
        os.system('cls')
        print("Affichage : " + str(int((x / n) * 100)) + "%")
    image.save(str(hmin) + 'x' + str(seuil) + 'x' + str(hmax) + 'x' + str(int(random() * (10 ** 3))) + '.png')
    return True

def printMatGrad(mat, couleurs = "gradient.png"):
    """Retourne la matrice de couleurs avec le gradient en entrée."""
    (matrice, min, max) = mat
    n = len(matrice)
    image = Image.new('RGB', (n, n))
    gradient = Image.open(couleurs)
    lGrad = gradient.size[0]
    for x in range(n):
        for y in range(n):
            image.putpixel((x, y), gradient.getpixel((map(matrice[x][y], (min, max), (0, lGrad - 1)), 1)) )
    image.save(str(int(random() * (10 ** 3))) + '.png')
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

def npIssou(height, width, coeffs, expo):
    """Tentative de vectorisation de la fonction Issou."""
    n = len(coeffs)
    simplex = OpenSimplex(seed = int(random() * (10 ** 15)))
    mat = []
    min = 100
    max = -1
    sum = 0
    for i in range(n):
        sum += coeffs[i]
    for y in range(0, height):
        mat.append([])
        for x in range(0, width):
            nx = x/width - 0.5
            ny = y/width - 0.5
            e = 0
            for i in range(n):
                e += abs(coeffs[i] * simplex.noise2d((2 ** i) * nx, (2 ** i) * ny))
            e = e / sum
            e = pow(e, expo)
            mat[y].append(e)
            if e < min:
                min = e
            elif e > max:
                max = e
    return (mat, min, max)

def salt(couleurs, gain):
    """Permet de 'salt' ie. d'ajouter de petites variations dans les couleurs de façon locale."""
    #(r, g, b) = couleurs #Couleurs de base en rgb
    #couleursS = np.sort(couleurs) #couleurs de base triées
    #dominant = couleursS[2]
    #if (dominant / (couleursS[1] + 0.1)) > 1.5:
    #    couleursN = [couleursS[0], couleursS[1], dominant + np.random.normal(0, gain)]
    #elif (couleurs[1] / (couleurs[0] + 0.1)) > 1.5:
    #    couleursN = [couleursS[0], couleursS[1] + np.random.normal(0, gain), dominant + np.random.normal(0, gain)]
    #else:
    #    couleursN = [couleurs[0] + np.random.normal(0, gain), couleurs[1] + np.random.normal(0, gain), dominant + np.random.normal(0, gain)]
    #Couleurs triées et modifiées
    #result = [0, 0, 0]
    result = []
    for i in range(3):
        result.append(int(couleurs[i] + np.random.normal(0, gain)))
    return tuple(result)
