import numpy as np
from PIL import Image
from usuelles import *
from TestPerlin import *


def SquareDiamond(taille, ecart, poids, nb):
    """Retourne une matrice de taille * taille en utilisant l'algorithme Square-Diamond avec un écart-type ecart et un poids poids sur le coeff rando. nb est le nombre d'itérations avec coeff aléatoire, le reste étant juste de l'interpolation"""
    matrice = np.zeros((taille, taille))
    matrice[0, 0] = poids * np.random.normal(scale = ecart)
    matrice[0, taille - 1] = poids * np.random.normal(scale = ecart)
    matrice[taille - 1, 0] = poids * np.random.normal(scale = ecart)
    matrice[taille - 1, taille - 1] = poids * np.random.normal(scale = ecart)
    padding = taille - 1
    while padding > 1:
        pas = padding // 2
        print(padding, pas)
        if padding % 1 != 0:
            print("Problème de dimension")
            return 0
        #Diamond
        for x in range(pas, taille, padding):
            for y in range(pas, taille, padding):
                if nb > 0 :
                    matrice[x, y] = (matrice[x - pas, y - pas] + matrice[x - pas, y + pas] + matrice[x + pas, y - pas] + matrice[x + pas, y + pas]) / 4 + poids * np.random.normal(scale = ecart)
                else :
                    matrice[x, y] = (matrice[x - pas, y - pas] + matrice[x - pas, y + pas] + matrice[x + pas, y - pas] + matrice[x + pas, y + pas]) / 4
        decalage = 0
        for x in range(0, taille, pas):
            if decalage == 0:
                decalage = pas
            else:
                decalage = 0
            for y in range(decalage, taille, padding):
                tot = 0
                n = 0
                if x >= pas:
                    tot += matrice[x - pas, y]
                    n += 1
                if x + pas < taille:
                    tot += matrice[x + pas, y]
                    n += 1
                if y >= pas:
                    tot += matrice[x, y - pas]
                    n += 1
                if y + pas < taille:
                    tot += matrice[x, y + pas]
                if nb > 0 :
                    matrice[x, y] = tot / n + poids * np.random.normal(scale = ecart)
                else :
                    matrice[x, y] = tot / n
        padding = pas
        nb -= 1
    return matrice

def printMatGrad(mat, couleurs = "gradient.png"):
    """Retourne la matrice de couleurs avec le gradient en entrée."""
    (matrice, mini, maxi) = mat, np.min(mat), np.max(mat)
    print(mini, maxi)
    n = len(matrice)
    image = Image.new('RGB', (n, n))
    gradient = Image.open(couleurs)
    lGrad = gradient.size[0]
    for x in range(n):
        print(x)
        for y in range(n):
            image.putpixel((x, y), gradient.getpixel( (int( map(matrice[x, y], (mini, maxi), (0, lGrad - 1)) ), 1)) )
    nom = str(np.random.randint(0, 99999))
    print(nom)
    image.save(nom + '.png')
    return True

def interpMatrix(matrice, Taillei, Taillef, methode = 'linear'):
    """Faire l'interpolation de la matrice."""
    if methode == 'linear' :
        interp = lambda t : t
    elif methode == 'polynome':
        interp = lambda t : 6 * (t ** 5) - 12 * (t ** 4) + 10 * (t ** 3)
        #interp = lambda t : t ** 2
    else:
        print('Erreur de méthode')
        return 0
    nombre = (Taillef - Taillei) / (Taillei - 1)
    if int(nombre) != nombre:
        print('Erreur de format')
        return 0
    nombre = int(nombre)
    facteur = 1 + nombre
    print(nombre, facteur)
    res = np.zeros((Taillef, Taillef))
    for x in range(Taillef):
        print('interp' + str(x))
        for y in range(Taillef):
            dx = x % facteur
            dy = y % facteur
            if dx == 0 and dy == 0:
                res[x, y] = matrice[x // facteur, y // facteur]
            else:
                qx = x // facteur
                qy = y // facteur
                cx = dx / facteur
                cy = dy / facteur
                ex = 1 - cx
                ey = 1 - cy
                if dx == 0:
                    res[x, y] = matrice[qx, qy] * interp(ey) + matrice[qx, qy + 1] * interp(cy)
                elif dy == 0:
                    res[x, y] = matrice[qx, qy] * interp(ex) + matrice[qx + 1, qy] * interp(cx)
                else:
                    res[x, y] = matrice[qx, qy] * (interp(ex * ey)) + matrice[qx + 1, qy + 1] * (interp(cx * cy)) + matrice[qx, qy + 1] * (interp(ex * cy)) + matrice[qx + 1, qy] * (interp(cx * ey))
    #Remap sur [0, 1]
    mini = np.min(res)
    maxi = np.max(res)
    res = (res - mini) * ( 1 - 0) / (maxi - mini) + 0
    return res


def taillemini(nombre):
    return int(np.log2(nombre)) + 1


def ajouterPerlin(matrice, octaves, pers, poids):
    """Ajoute à la matrice un bruit de Perlin."""
    Taille = 2 ** taillemini(matrice.shape[0])
    bruit = generate_octaves((Taille, Taille), 2, pers, octaves)
    res = bruit[0:matrice.shape[0], 0:matrice.shape[0]]
    return matrice + (poids * res)
