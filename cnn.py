import numpy as np
from scipy import signal
import math
import skimage.measure
import copy
from PIL import Image
import os

def loadimage(nom):
    im = Image.open(nom).convert('L')
    return (np.array(im).flatten() / 255)

PERSISTANCE = 0.05

pathCroix = 'DATAIN\\CROIX\\'
pathRond = 'DATAIN\\ROND\\'

imgCROIX = []
imgROND = []

for _, _, f in os.walk(pathCroix):
    for issou in f:
        imgCROIX.append(np.reshape(loadimage(pathCroix + issou), (10, 10)))

for _, _, f in os.walk(pathRond):
    for issou in f:
        imgROND.append(np.reshape(loadimage(pathRond + issou), (10, 10)))

DATAIN = imgCROIX + imgROND
EXPECT = np.concatenate((np.full(len(imgCROIX), [1, 0], dtype = '2f'), np.full(len(imgROND), [0, 1], dtype = '2f')))

def reLu(x):
    return np.maximum(x, 0, x)

def sigmoid(x):
    try:
        ans = 1 / (1 + np.exp(-x))
    except OverflowError:
        ans = 0.5
        print('Overflow                   !!!!!!!')
    return ans

class inputLayer:
    def __init__(self, nom, tailleIn):
        self.data = np.zeros((tailleIn, tailleIn))
        self.name = nom

    def inputData(self, data):
        self.data = data

class convolutionLayer:
    def __init__(self, nom, tailleIn, tailleNoyau, nbFiltres): #tailleIn est la taille de l'image étudiée, tailleNoyau celle du noyau
        self.nom = nom
        self.nbFiltres = nbFiltres
        self.tailleIn = tailleIn
        self.tailleNoyau = tailleNoyau
        self.data = np.zeros((nbFiltres, tailleIn - tailleNoyau +  1, tailleIn - tailleNoyau +  1), dtype = 'float')
        self.filtres = np.random.rand(nbFiltres, tailleNoyau, tailleNoyau) * 2 - 1

    def propagate(self, dataIn):
        """Remplit le data de la couche en faisant les convolutions sur dataIn."""
        for i in range(self.nbFiltres):
            self.data[i] = reLu(signal.convolve(dataIn, self.filtres[i], mode = 'valid'))

    def backPropagate(self, dH, W, nbFiltres, DELTA, DATA):
        global PERSISTANCE
        """dH est le gradient des couches en aval, déjà calculé."""
        dX = np.zeros((self.tailleIn, self.tailleIn))
        dF = np.zeros((self.nbFiltres, self.tailleNoyau, self.tailleNoyau))
        temp = np.rot90(np.rot90(dH))
        nbFiltres, X, Y = dH.shape
        for i in range(self.nbFiltres):
            for x in range(X):
                for y in range(Y):
                    dX[x:x + self.tailleNoyau, y:y + self.tailleNoyau] += self.filtres[i] * temp[i, x, y]
                    dF[i] += DATA[x:x + self.tailleNoyau, y:y + self.tailleNoyau] * temp[i, x, y]
            self.filtres[i] -= PERSISTANCE * dF[i] * (self.filtres[i])
        return (0, 0, 0, DELTA)



class poolLayer:
    def __init__(self, nom, tailleIn, tailleNoyau = 2, mode = 'max'):
        self.nom = nom
        self.tailleIn = tailleIn
        self.mode = mode
        self.tailleNoyau = tailleNoyau
        self.data = None

    def propagate(self, data):
        """Propage sur la couche de pool, ie. fait un downsampling."""
        liste = []
        for dat in data:
            if self.mode == 'max':
                liste.append(skimage.measure.block_reduce(dat, (self.tailleNoyau, self.tailleNoyau), np.max))
            elif mode == 'meam':
                liste.append(skimage.measure.block_reduce(dat, (self.tailleNoyau, self.tailleNoyau), np.mean))
            elif mode == 'sum':
                liste.append(skimage.measure.block_reduce(dat, (self.tailleNoyau, self.tailleNoyau), np.sum))
        self.data = np.array(liste)

    def backPropagate(self, dH, W, nbFiltres, DELTA, DATA):
        dW = np.zeros((dH.shape[0] * self.tailleNoyau, dH.shape[1] * self.tailleNoyau, dH.shape[2] * self.tailleNoyau))
        for i in range(nbFiltres):
            for x in range(dW.shape[1]):
                for y in range(dW.shape[2]):
                    dW[x, y] = dH[i, x // self.tailleNoyau, y // self.tailleNoyau]
        return dW, W, nbFiltres, DELTA

class fullyConnected:
    def __init__(self, nom, tailleIn, taille, type = 'hidden'):
        """taille est la taille de la couche, tailleIn celle de la couche précédente."""
        """type peut être 'hidden', 'junction' ou 'output'."""
        self.nom = nom
        self.tailleIn = tailleIn
        self.taille = taille
        self.weights = np.random.rand(self.taille, self.tailleIn) * 2 - 1 #Les poids sont ceux pour venir à la matrice!
        self.data = np.zeros(self.taille)
        self.type = type

    def propagate(self, data):
        if self.type == 'junction':
            self.data = np.ndarray.flatten(data)
        else:
            self.data = sigmoid(np.dot(data, self.weights.T))

    def outputData(self):
        return self.data

    def backPropagate(self, dH, W, nbFiltres, DELTA, DATA):
        """W est la matrice de poids associés au passage à la couche suivante."""
        global PERSISTANCE
        if self.type == 'junction':
            dX = np.array(self.tailleIn)
            dW = np.array(W.shape)
            der = np.expand_dims(self.data, 0) * (1 - np.expand_dims(self.data, 0))
            dW = np.dot(dH, W) * der
            tailleAvant = int(np.sqrt(self.tailleIn // nbFiltres))
            return (np.reshape(dW, (nbFiltres, tailleAvant, tailleAvant)), W, nbFiltres, DELTA)
        else:
            dX = np.array(self.tailleIn)
            dW = np.array(W.shape)
            der = np.expand_dims(self.data, 0) * (1 - np.expand_dims(self.data, 0))
            dW = np.dot(dH, W) * der
            #self.weights -= PERSISTANCE * np.dot(self.data, dW)
            DELTA.append(np.dot(self.data, dW.T))
            return(dW, self.weights, nbFiltres, DELTA)

class network:
    def __init__(self, nom, layers):
        self.nom = nom
        self.layers = layers
        self.layersNumber = len(layers)
        self.epochs = 0 #Le nombre de cycles de retropropagation deja faits

    def propagateLayers(self):
        for i in range(1, self.layersNumber):
            self.layers[i].propagate(self.layers[i - 1].data)

    def inputData(self, data):
        self.layers[0].inputData(data)

    def outputData(self):
        return self.layers[self.layersNumber - 1].data

    def lossOne(self, data, expect):
        self.inputData(data)
        self.propagateLayers()
        resultat = self.outputData()
        ret = 0
        for i in range(len(expect)):
            ret += (expect[i] - resultat[i]) ** 2
        return ret

    def lossAll(self, DATAIN, EXPECT):
        loss = 0
        for i in range(len(DATAIN)):
            loss += self.lossOne(DATAIN[i], EXPECT[i])
        return loss

    def train(self, DATAIN, EXPECT, number):
        for j in range(number):
            print(j)
            print(self.lossAll(DATAIN, EXPECT))
            for i in range(len(DATAIN)):
                self.inputData(DATAIN[i])
                self.propagateLayers()
                self.backPropagateLayers(EXPECT[i], DATAIN[i])
            self.epochs += 1

    def backPropagateLayers(self, expect, DATAIN):
        global PERSISTANCE
        i = self.layersNumber - 1
        data = self.layers[i].data
        diff = data - expect
        dH = diff * np.expand_dims(data, 0) * (1 - np.expand_dims(data, 0))
        W = self.layers[i].weights
        nbFiltres = self.layers[1].nbFiltres
        DELTA = []
        for i in reversed(range(1, self.layersNumber - 1)):
            dH, W, nbFiltres, DELTA = self.layers[i].backPropagate(dH, W, nbFiltres, DELTA, DATAIN)

        #Application sur les fullyConnected
        for i in range(len(DELTA)):
            self.layers[self.layersNumber - i - 1].weights -= PERSISTANCE * np.dot(np.expand_dims(self.layers[self.layersNumber - i - 2].data, axis = 0).T, np.expand_dims(DELTA[i], axis = 0)).T


input = inputLayer('input', 10)
conv = convolutionLayer('conv', 10, 3, 16)
po = poolLayer('po', 8)
trans = fullyConnected('trans', 256, 256, type = 'junction')
f2 = fullyConnected('f2', 256, 64)
f25 = fullyConnected('f25', 64, 16)
f3 = fullyConnected('f3', 16, 2, type = 'output')
net = network('net', [input, conv, po, trans, f2, f25, f3])
