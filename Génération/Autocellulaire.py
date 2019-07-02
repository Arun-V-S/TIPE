import numpy as np
import random as rand
from PIL import Image
import os
from usuelles import * #Importe les fonctions usuelles

RANDSET = [5, 250]
RAND = 2


def randinit(nombre, bornesX, bornesY, bornesZ, marge):
    """Renvoie nombre triplets de points au hasard dans le carré donné avec des marges autour des bords, à valeur dans (zmin, zmax)."""
    global RANDSET
    SET = False #On choisit dans le set ou alros au hasard totalement?
    res = []
    (xmin, xmax) = bornesX
    (ymin, ymax) = bornesY
    (zmin, zmax) = bornesZ
    if SET:
        for _ in range(nombre):
            x = rand.randint(xmin + marge, xmax - marge)
            y = rand.randint(ymin + marge, ymax - marge)
            z = RANDSET[rand.randint(0, 1)]
            res.append((x, y, z))
    else:
        for _ in range(nombre):
            rand.seed()
            x = rand.randint(xmin + marge, xmax - marge)
            y = rand.randint(ymin + marge, ymax - marge)
            z = rand.randint(zmin, zmax)
            res.append((x, y, z))
    return list(set(res))

def addTuple(t1, t2):
    """Additionne les 2 tuples composante à composante."""
    if len(t1) != len(t2):
        return False
    else:
        return tuple(t1[i] + t2[i] for i in range(len(t1)))

def inMargins(X, Y, marginX, marginY):
    return (marginX[0] <= X and X <= marginX[1]) and (marginY[0] <= Y and Y <= marginY[1])

def DEP(n, k = 1):
    """Renvoie le DEP pour des voisins de rang n, en interpolant de k pixels."""
    res = []
    for i in range(- n, n + 1):
        for j in range(- n, n + 1):
            res.append((i * k, j * k))
    return res

DEP1 = [(1, 0), (-1, 0), (0, 1), (0, -1)] #Voisins immédiats
DEP2 = [(1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, -1), (0, -1), (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2)] #Voisins à 2 degrés
#DEP2 = [(range(-3, 4), range(-3, 4))]
DEP3 = DEP(3)
DEP10 = DEP(10, 3)
DEP4 = DEP(4, 1)


def voisins(point, tailleX, tailleY, DEP):
    """Retourne les points voisins du point dans la matrice grâce à DEP."""
    res = []
    (X, Y, _) = point
    for dep in DEP:
        (ptX, ptY) = addTuple(dep, (X, Y))
        if inMargins(ptX, ptY, (0, tailleX - 1), (0, tailleY - 1)):
            res.append((ptX, ptY, -1))
    return res

def pondere(matrice, point, DEP):
    global RAND
    tailleX, tailleY = len(matrice[0]), len(matrice)
    (X, Y, _) = point
    Z = 0
    n = 0
    for dep in DEP:
        (ptX, ptY) = addTuple(dep, (X, Y))
        if inMargins(ptX, ptY, (0, tailleX - 1), (0, tailleY - 1)):
            z = matrice[ptY][ptX]
            if z != -1:
                Z += z
                n += 1
    if n != 0:
        return ((Z // n) + rand.randint(-RAND, RAND), n)
    else:
        return []


def generation(taille, gradientIm, nInit, marge):
    """Génère une image aléatoire à partir d'unt matrice de taille taille, initialisée avec nInit points, avec gradientIm pour le rendu."""
    minimum = 255
    maximum = 0
    POINTS = []
    tailleX, tailleY = taille
    NTOT = tailleX * tailleY
    matrice = [[(- 1) for _ in range(tailleX)] for _ in range(tailleY)]
    pointsInit = randinit(nInit, (0, tailleX), (0, tailleY), (0, 255), marge)
    n = len(pointsInit)
    for pointCourant in pointsInit:
        (x, y, z) = pointCourant
        matrice[y][x] = z
        POINTS += voisins(pointCourant, tailleX, tailleY, DEP4)
    while n < NTOT:
        POINTSTEMP = []
        POINTS = list(set(POINTS))
        for pointcourant in POINTS:
            (X, Y, _) = pointcourant
            Pondere = pondere(matrice, pointcourant, DEP2)
            #print(len(Pondere), matrice[Y][X])
            if len(Pondere) != 0 and matrice[Y][X] == -1:
                matrice[Y][X] = Pondere[0]
                Temp = Pondere[0]
                if Temp < minimum:
                    minimum = Temp
                elif Temp > maximum:
                    maximum = Temp
                n += 1
                if n > NTOT:
                    return matrice
                if n % 100 == 0:
                    print(n, NTOT)
            POINTSTEMP += voisins(pointcourant, tailleX, tailleY, DEP1)
        POINTS = [] + POINTSTEMP
        POINTSTEMP = []
    return matrice#, minimum, maximum


def affichage(matrice):
    """Affiche en niveaux de gris l'image associée à la matrice matrice."""
    tailleX, tailleY = len(matrice[0]), len(matrice)
    img = Image.new('L', (tailleX, tailleY))
    pixel = img.load()
    for i in range(tailleX):
        for j in range(tailleY):
            if matrice[j][i] == -1:
                matrice[j][i] = 0
            pixel[i, j] = matrice[j][i]
    img.show()

def printMatGrad(mat, couleurs = "gradient.png"):
    """Retourne la matrice de couleurs avec le gradient en entrée."""
    matrice, min, max = mat, 0, 255
    n = len(matrice)
    image = Image.new('RGB', (n, n))
    gradient = Image.open(couleurs)
    lGrad = gradient.size[0]
    for x in range(n):
        for y in range(n):
            valeur = abs(matrice[x][y])
            coordonnees = (map(valeur, (min, max), (0, lGrad - 2)), 1)
            couleur = gradient.getpixel(coordonnees)
            image.putpixel((x, y), couleur)
    image.save(os.path.join("Outputs", str(int(rand.random() * (10 ** 3))) + '.png'))
    return True
