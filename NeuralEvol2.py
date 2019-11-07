## Algorithme d'évolution de réseaux de neurones
## Ces réseaux sont représentés sous la forme d'un graphe orienté
import matplotlib.pyplot as plt
import numpy as np
import copy
import networkx as nx
from math import *
from PIL import Image
import os

##Hyperparamètres
ALPHA = 0.1
TAILLE = 25 #nombre de couches virtuelles des réseaux
#Mutation
WEIGHTMUTATION = 0.5 #Correcteur de mutabilité des poids p/r à la mutabilité standard du réseau
EDGEMUTATION = 0.1 #Correcteur de mutabilité pour l'apparition d'un lien
VERTEXMUTATION = 0.025 #Correcteur de mutabilité pour l'apparition d'un neurone
EPSILON = 0.0001 #Mutationrate minimal
RAFRAICHISSEMENT = 0.5 #Probabilité de rajouter des réseaux vierges lors de l'écrêmage
ITERATIONS = 1


def getLetter(index):
    assert 0 <= index
    return chr(index + 65)

class Activation:
    def __init__(self, function, name):
        self.function = function
        self.name = name

    def __call__(self, x):
        try:
            return self.function(x)
        except:
            print('OVERFLOW')
            return self.function(0)

    def __repr__(self):
        return self.name

leakyReLU = Activation(lambda x : x if x >= 0 else ALPHA * x, "leakyReLU")
Sigmoid = Activation(lambda x : 1 / (1 + exp(-x)), "Sigmoid")
fonctionRandom = lambda : np.random.normal()
flatten = lambda l: [item for sublist in l for item in sublist]

class Dataloader:
    def __init__(self, DATAIN, EXPECT):
        assert len(DATAIN) == len(EXPECT), "Pb de correspondance entre DATAIN et EXPECT"
        self.datain = DATAIN
        self.expect = EXPECT
        self.longueur = len(DATAIN)

    def __getitem__(self, index):
        assert index < self.longueur, "Dépassement d'indice"
        return (self.datain[index], self.expect[index])

class Vertex:
    def __init__(self, number, activation, layer, vertexType = 'inner'):
        self.activation = activation
        self.number = number
        self.data = None #Pas de récurrence donc data unique!
        self.bias = None
        self.weights = []
        self.voisins = [] # Il s'agit des voisins en amont!
        self.bias = 0
        self.vertexType = vertexType
        self.layer = layer

    def __repr__(self):
        return getLetter(self.number)

class Network:
    def __init__(self, inputNumber, outputNumber, activationList):
        self.mutationRate = np.random.random()
        vertexNumber = inputNumber + outputNumber
        assert vertexNumber == len(activationList), 'Probleme des activations'
        self.vertexNumber = vertexNumber
        self.inputNumber = inputNumber
        self.outputNumber = outputNumber
        self.vertex = [[] for _ in range(TAILLE)]
        self.vertexList = []
        for i in range(inputNumber):
            N = Vertex(i, activationList[i], 0, 'input')
            self.vertex[0].append(N)
            self.vertexList.append(N)
        for i in range(inputNumber, inputNumber + outputNumber):
            N = Vertex(i, activationList[i], TAILLE - 1, 'output')
            self.vertex[TAILLE - 1].append(N)
            self.vertexList.append(N)
        self.fitness = 0

    def initRandom(self, functionData, functionWeights, functionBias):
        #Remplir les neurones avec les fonctions (éventuellement aléatoires)
        for neurone in self.vertexList:
            longueur = len(neurone.voisins)
            for i in range(longueur):
                neurone.weights[i] = functionWeights()
            neurone.data = functionData()
            neurone.bias = functionBias()

    def feedForward(self):
        for layer in range(1, TAILLE):
            for N in self.vertex[layer]:
                somme = N.bias
                for i in range(len(N.voisins)):
                    somme += N.weights[i] * self.vertexList[N.voisins[i]].data
                N.data = N.activation(somme)


    def feedIn(self, values):
        assert len(values) == self.inputNumber, 'Probleme de concordance des données d entrée.'
        for i in range(len(values)):
            self.vertex[0][i].data = values[i]

    def feedOut(self): #Donner les valeurs des neurones de sortie
        values = []
        for i in range(self.outputNumber):
            values.append(self.vertex[TAILLE - 1][i].data)
        return values

    def addEdge(self, i, j, weight = 0): #Ajouter une arête au réseau
        assert not i in self.vertex[j].voisins, 'Arête déjà présente.'
        if self.vertexList[i].layer < self.vertexList[j].layer:
            self.vertex[j].voisins.append(i)
            self.vertex[j].weights.append(weight)

    def mutate(self):
        #Mutation des poids
        for V in self.vertexList:
            for i in range(len(V.weights)):
                if np.random.random() <= self.mutationRate * WEIGHTMUTATION:
                    V.weights[i] += np.random.normal(0, WEIGHTMUTATION)
        #Mutation des neurones
        while np.random.random() <= self.mutationRate * VERTEXMUTATION:
            self.addRandomEdge()

        while np.random.random() <= self.mutationRate * EDGEMUTATION:
            self.addRandomEdgeWithoutVertex()

        #Mutation du taux de mutation
        if np.random.random() <= self.mutationRate:
            self.mutationRate = max(min(EPSILON, self.mutationRate + np.random.normal(0, 0.1)), 1 - EPSILON)

    def addRandomEdgeWithoutVertex(self):
        (i, j) = np.random.randint(0, self.vertexNumber, 2)
        coucheI = self.vertexList[i].layer
        coucheJ = self.vertexList[j].layer
        if coucheI == coucheJ:
            return False
        elif coucheI > coucheJ:
            i, j = j, i

        if i not in self.vertexList[j].voisins:
            self.vertexList[j].voisins.append(i)
            self.vertexList[j].weights.append(np.random.normal())

    def addRandomEdge(self):
        (i, j) = np.random.randint(0, self.vertexNumber, 2)
        coucheI = self.vertexList[i].layer
        coucheJ = self.vertexList[j].layer
        if abs(coucheI - coucheJ) <= 1:
            return False
        if coucheI > coucheJ:
            i, j = j, i
            coucheI, coucheJ = coucheJ, coucheI

        layerNew = np.random.randint(coucheI + 1, coucheJ)

        if i in self.vertexList[j].voisins:
            indice = self.vertexNumber
            E = Vertex(indice, leakyReLU if np.random.random() < 0.5 else Sigmoid, layerNew, vertexType = 'inner')
            E.voisins = [i]
            E.data = 0
            E.weights = [np.random.normal()]
            self.vertexList.append(E)
            self.vertex[layerNew].append(E)
            self.vertexNumber += 1
            self.vertexList[j].voisins = [indice if x == i else x for x in self.vertexList[j].voisins]
        else:
            indice = self.vertexNumber
            E = Vertex(indice, leakyReLU if np.random.random() < 0.5 else Sigmoid, layerNew, vertexType = 'inner')
            E.voisins = [i]
            E.data = 0
            E.weights = [np.random.normal()]
            self.vertexList.append(E)
            self.vertex[layerNew].append(E)
            self.vertexNumber += 1
            self.vertexList[j].voisins.append(indice)
            self.vertexList[j].weights.append(np.random.normal())

    def evaluate(self, dataloader):
        somme = 0
        sommepartielle = 0
        for i in range(dataloader.longueur):
            inD, outD = dataloader[i]
            self.feedIn(inD)
            self.feedForward()
            res = self.feedOut()
            for j in range(len(res)):
                sommepartielle += (res[j] - outD[j]) ** 2
            somme += sommepartielle
        self.fitness = somme / dataloader.longueur

    def test(self, datain):
        self.feedIn(datain)
        self.feedForward()
        return self.feedOut()

class Pool:
    def __init__(self, networkNumber, dataloader, *args):
        self.dataloader = dataloader
        self.dataSize = dataloader.longueur
        self.networkNumber = networkNumber
        self.population = [Network(*args) for i in range(networkNumber)]
        for N in self.population:
            N.initRandom(fonctionRandom, fonctionRandom, fonctionRandom)

    def initRandom(self, *args):
        for N in self.population:
            try:
                N.initRandom(args)
            except:
                print("Problème d'initialisation : fonctions d'init invalides")

    def mutateNetwork(self):
        for N in self.population:
            N.mutate()

    def evaluate(self):
        for N in self.population:
            N.evaluate(dataloader)
        self.population.sort(key = lambda object : object.fitness)

    def separate(self, keep):
        keptPopulation = self.population[0:keep]
        for i in range(keep, self.networkNumber):
            indice = int(min(abs(np.random.normal()), 1) * (keep - 1))
            N = copy.deepcopy(keptPopulation[indice])
            N.mutate()
            keptPopulation.append(N)
        self.population = keptPopulation

    def train(self, epochNumber, keep):
        assert keep <= self.networkNumber
        for i in range(epochNumber):
            self.evaluate()
            self.separate(keep)
            print(str(i) + ' ' + str(self.population[0].fitness))
            if self.population[0].fitness < EPSILON:
                break

    def renormalize(self):
        for N in self.population:
            N.initRandom(fonctionRandom, fonctionRandom, fonctionRandom)

"""
DATAIN = [[0, 0], [1, 0], [0, 1], [1, 1]]
EXPECT = [[0], [1], [1], [0]]
dataloader = Dataloader(DATAIN, EXPECT)
"""

"""
network = Network(4, 1, 1, [leakyReLU, leakyReLU, leakyReLU, leakyReLU])
network.addEdge(0, 2)
network.addEdge(2, 3)
network.addEdge(3, 1)
network.addEdge(2, 2)
network.addEdge(0, 0)
network.initRandom(fonctionRandom, fonctionRandom, fonctionRandom)
for i in range(10):
    network.feedIn([1])
    network.feedForward()
    print(network.feedOut())
    #print(network.vertex[3].dataOut)
"""

def plotBestOld(i = 0):
    G = nx.DiGraph()
    P.evaluate()
    N = P.population[i]
    number = N.vertexNumber
    input = N.inputNumber
    output = N.outputNumber
    color_map = []
    for i in range(number):
        G.add_node(getLetter(i))
        if i < input:
            color_map.append('red')
        elif i < input + output:
            color_map.append('green')
        else:
            color_map.append('blue')
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color = color_map)
    for i in range(number):
        voisins = N.vertexList[i].voisins
        weights = N.vertexList[i].weights
        for j in range(len(voisins)):
            G.add_edge(getLetter(voisins[j]), getLetter(i), weight = weights[j], arrowstyle='->')
    for (u, v, d) in G.edges(data = True):
        nx.draw_networkx_edges(G, pos, edgelist = [(u, v)], alpha = Sigmoid(abs(d['weight'])), arrowstyle='fancy', arows = True)
    #Dessin:
    #nx.draw_networkx(G, with_labels = True, arrowstyle='fancy')
    plt.show()


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
        imgCROIX.append(np.reshape(loadimage(pathCroix + issou), (100)))

for _, _, f in os.walk(pathRond):
    for issou in f:
        imgROND.append(np.reshape(loadimage(pathRond + issou), (100)))

DATAIN = imgCROIX + imgROND
EXPECT = np.concatenate((np.full(len(imgCROIX), [1, 0], dtype = '2f'), np.full(len(imgROND), [0, 1], dtype = '2f')))

dataloader = Dataloader(DATAIN, EXPECT)

P = Pool(250, dataloader, 100, 1, [Sigmoid for _ in range(101)])
