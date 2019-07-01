import numpy as np
from PIL import Image

def convertIm(fichier):
    """Convertit l'image 'fichier' en une matrice, en nuances de gris."""
    image = Image.open(fichier)
    image = image.convert('L')
    (x, y) = image.size
    matrice = [[0 for j in range(x)] for i in range(y)]
    for i in range(x): 
        for j in range(y):
            matrice[j][i] = image.getpixel((i, j))
    return matrice


def gradient(matrice):
    """Retourne une matrice de gradient des valeurs de matrice."""
    height = len(matrice)
    width = len(matrice[0])
    res = np.zeros(width * height, dtype=int).reshape(height, width)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            WE = abs(matrice[y][x + 1] - matrice[y][x - 1])
            NS = abs(matrice[y - 1][x] - matrice[y + 1][x])
            res[y][x] = WE + NS
    return res

def filtrePH(valeur, coup, val1, val2):
    """Applique un filtre passe-haut sur valeur en fonction de coup."""
    if valeur < coup:
        return val1
    else:
        return val2

def affichage(matrice, nomimage):
    """Affiche l'image associée à matrice en N&B."""
    x = len(matrice[0])
    y = len(matrice)
    image = Image.new('L', (x, y))
    for i in range(1, x - 1):
        for j in range(1, y - 1):
            image.putpixel((i, j), (255 - filtrePH(matrice[j][i], 80, 0, 255),))
    image.save('Grad' + nomimage)

def gradientIm(image):
    affichage(gradient(convertIm(image)), image)
    return True
