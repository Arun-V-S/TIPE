##Application de matrices de convolution sur des images
import numpy as np
import random as rand
from PIL import Image
import os
from usuelles import * #Importe les fonctions usuelles
from scipy import signal

IDENTITE = (np.array([[0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 0]]),
                      1, 'IDENTITE')

GRADIENT = (np.array([[0, 1, 0],
                      [1, 0, -1],
                      [0, -1, 0]]),
                      1, 'GRADIENT')

NET = ([[0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]],
        1, 'NET')

GAUSSIEN = (np.array([[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 2]]),
                      16, 'GAUSSIEN')

CONTRASTE = (np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]]),
                       1, 'CONTRASTE')

BORDS = (np.array([[0, 0, 0],
                   [-1, 0, 0],
                   [0, 0, 0]]),
                   1, 'BORDS')



def convolSci(nomImage, matriceConvol):
    mode = 'same'
    matrice, diviseur, NOM = matriceConvol
    image = np.asarray(Image.open(nomImage))
    I1 = image[:, :, 0]
    I2 = image[:, :, 1]
    I3 = image[:, :, 2]
    R = signal.convolve(I1, matrice, mode)
    G = signal.convolve(I2, matrice, mode)
    B = signal.convolve(I3, matrice, mode)
    R = np.absolute(R) // diviseur
    G = np.absolute(G) // diviseur
    B = np.absolute(B) // diviseur
    (h, l) = I1.shape
    G = G[:, :, np.newaxis]
    R = R[:, :, np.newaxis]
    B = B[:, :, np.newaxis]
    res = np.concatenate((R, G, B), axis = 2)
    res = np.uint8(res)
    image = Image.fromarray(res)
    image.save(NOM + ' ' + nomImage)
    #R = signal.convolve(I1, matrice, mode)
    #R = 255.*np.absolute(R)/np.max(R)
    #R = Image.fromarray(R, 'L')
    #G = Image.fromarray(G, 'L')
    #B = Image.fromarray(B, 'L')

    #res = Image.merge('RGB', (R, G, B))
    #.save('LOL.png')
    #res.save('Isssssssssssssou.png')



def lectureImage(nomImage, couleur = True):
    image = Image.open(nomImage)
    if not couleur :
        image = image.Image.convert(mode = 'L')
    tailleX, tailleY = image.size
    if couleur :
        matrice = np.zeros((tailleY, tailleX), dtype=(int,3))
    else:
        matrice = np.zeros((tailleY, tailleX), dtype=(int,1))
    for i in range(tailleY):
        for j in range(tailleX):
            matrice[i, j] = image.getpixel((j, i))
    return (matrice, tailleX, tailleY)



def convolImage(nomImage, matriceConvoTuple, couleur = True):
    #matrice, tailleX, tailleY = lectureImage(nomImage, couleur)
    image = Image.open(nomImage)
    tailleX, tailleY = image.size
    matrice = np.array(image)
    resultat = np.empty((tailleX, tailleY), dtype=(int,3))
    matriceConvo, diviseur, NOM = matriceConvoTuple
    #patch = [] #Futur morceau de la matrice qui sera envoyé à la fonction convolution
    if couleur:
        mode = 'RGB'
    else:
        mode = 'L'
    #resultat = Image.new(mode, (tailleX, tailleY))
    for i in range(tailleX):
        print(int(100 * (i / tailleX)))
        for j in range(tailleY):
            patch = np.zeros((3, 3), dtype=(int,3))
            for a in range(-1, 2): #Crée le patch, avec -1 si on est en dehors des limites de la matrice
                for b in range(-1, 2):
                    if (b + j) >= 0 and (b + j) < tailleY and (a + i) >= 0 and (a + i) < tailleX:
                        patch[a + 1, b + 1] = tuple(matrice[j + b, i + a])
                    else:
                        patch[a + 1, b + 1] = (-1, -1, -1)
            resultat[i, j] = convolution(patch, matriceConvo, couleur, diviseur)
    print(resultat)
    imgRes = Image.fromarray(resultat)
    img.res.save('convolu-' + nomImage)
            #resultat.putpixel((i, j), convolution(patch, matriceConvo, couleur, diviseur))
    resultat.save(NOM + nomImage)
    return


def convolution(patch, matriceConvo, couleur, diviseur):
    """Retourne la valeur du point (x, y) en fonction du patch et de la matrice de convolution choisie."""
    #patch = patch[:, 0]
    if couleur:
        resR, resG, resB = 0, 0, 0
        for i in range(3):
            for j in range(3):
                pixelCourant = patch[i, j]
                #print(pixelCourant)
                #print(pixelCourant)
                if not np.any(pixelCourant == (-1, -1, -1)) : #patch[i, j] != np.array([-1, -1, -1])
                    resR += patch[i, j][0] * matriceConvo[i][j]
                    resG += patch[i, j][1] * matriceConvo[i][j]
                    resB += patch[i, j][2] * matriceConvo[i][j]
        return tuple([int(resR / diviseur), int(resG / diviseur), int(resB / diviseur)])
    else:
        res = 0
        for i in range(3):
            for j in range(3):
                res += patch[i, j][0] * matriceConvo[i][j]
        return (int(res / diviseur))

convolutionVect = np.vectorize(convolution)
