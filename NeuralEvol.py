## Algorithme d'évolution de réseaux de neurones
## Ces réseaux sont représentés sous la forme d'un graphe orienté
import matplotlib.pyplot as plt
import numpy as np
import copy
import networkx as nx
from math import *

##Hyperparamètres
ALPHA = 0.1
#Mutation
WEIGHTMUTATION = 0.5 #Correcteur de mutabilité des poids p/r à la mutabilité standard du réseau
EDGEMUTATION = 0.025 #Correcteur de mutabilité pour l'apparition d'un neurone
VERTEXMUTATION = 0.1
EPSILON = 0.0001 #Mutationrate minimal
RAFRAICHISSEMENT = 0.5 #Probabilité de rajouter des réseaux vierges lors de l'écrêmage
ITERATIONS = 5


def getLetter(index):
    assert 0 <= index
    return chr(index + 65)

class Activation:
    def __init__(self, function):
        self.function = function

    def __call__(self, x):
        try:
            return self.function(x)
        except:
            print('OVERFLOW')
            return self.function(0)

leakyReLU = Activation(lambda x : x if x >= 0 else ALPHA * x)
Sigmoid = Activation(lambda x : 1 / (1 + exp(-x)))
fonctionRandom = lambda : np.random.normal()

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
    def __init__(self, number, activation, vertexType = 'inner'):
        self.activation = activation
        self.number = number
        self.dataIn = None #Besoin de différencier les deux pour le cas de récurrences,
        self.dataOut = None #permettant de bien faire avancer le réseau par paliers
        self.bias = None
        self.weights = []
        self.voisins = [] # Il s'agit des voisins en amont!
        self.bias = 0
        self.vertexType = vertexType

    def __repr__(self):
        return str(self.number)

    def feedDataIntern(self):
        self.dataOut = self.dataIn

class Network:
    def __init__(self, vertexNumber, inputNumber, outputNumber, activationList):
        self.mutationRate = np.random.random()
        assert inputNumber + outputNumber <= vertexNumber, 'Probleme de nombre de neurones'
        assert vertexNumber == len(activationList), 'Probleme des activations'
        self.vertexNumber = vertexNumber
        self.inputNumber = inputNumber
        self.outputNumber = outputNumber
        self.inputvertex = [i for i in range(inputNumber)]
        self.outputvertex = [i for i in range(inputNumber, inputNumber + outputNumber)]
        self.vertex = []
        for i in range(inputNumber):
            self.vertex.append(Vertex(i, activationList[i], 'input'))
        for i in range(inputNumber, inputNumber + outputNumber):
            self.vertex.append(Vertex(i, activationList[i], 'output'))
        for i in range(inputNumber + outputNumber, vertexNumber):
            self.vertex.append(Vertex(i, activationList[i]))
        self.fitness = 0

    def initRandom(self, functionData, functionWeights, functionBias):
        #Remplir les neurones avec les fonctions (éventuellement aléatoires)
        for neurone in self.vertex:
            longueur = len(neurone.voisins)
            for i in range(longueur):
                neurone.weights[i] = functionWeights()
            neurone.dataOut = functionData()
            neurone.bias = functionBias()

    def feedInternal(self):
        #Fait passer les valeurs In à Out sur tous les neurones
        for neurone in self.vertex:
            neurone.feedDataIntern()

    def feedForward(self):
        for neurone in self.vertex:
            somme = neurone.bias
            for i in range(len(neurone.voisins)):
                somme += neurone.weights[i] * self.vertex[neurone.voisins[i]].dataOut
            neurone.dataIn = neurone.activation(somme)
        self.feedInternal()

    def feedIn(self, values):
        assert len(values) == self.inputNumber, 'Probleme de concordance des données d entrée.'
        for i in range(len(values)):
            self.vertex[self.inputvertex[i]].dataIn = values[i]
            self.vertex[self.inputvertex[i]].dataOut = values[i]
        #self.feedInternal()

    def feedOut(self): #Donner les valeurs des neurones de sortie
        values = []
        for i in range(self.outputNumber):
            values.append(self.vertex[self.outputvertex[i]].dataOut)
        return values

    def addEdge(self, i, j, weight = None): #Ajouter une arête au réseau
        assert not i in self.vertex[j].voisins, 'Arête déjà présente.'
        self.vertex[j].voisins.append(i)
        self.vertex[j].weights.append(weight)

    def mutate(self):
        #Mutation des poids
        for V in self.vertex:
            for i in range(len(V.weights)):
                if np.random.random() <= self.mutationRate * WEIGHTMUTATION:
                    V.weights[i] += np.random.normal(0, WEIGHTMUTATION)
        #Mutation des neurones
        while np.random.random() <= self.mutationRate * EDGEMUTATION:
            self.addRandomEdge()

        while np.random.random() <= self.mutationRate * VERTEXMUTATION:
            self.addRandomEdgeWithoutVertex()

        #Mutation du taux de mutation
        if np.random.random() <= self.mutationRate:
            self.mutationRate = max(min(EPSILON, self.mutationRate + np.random.normal(0, 0.1)), 1 - EPSILON)

    def addRandomEdgeWithoutVertex(self):
        (i, j) = np.random.randint(0, self.vertexNumber, 2)
        if i not in self.vertex[j].voisins and i != j and j not in range(self.inputNumber, self.inputNumber + self.outputNumber):
            self.vertex[i].voisins.append(i)
            self.vertex[i].weights.append(np.random.normal())




    def addRandomEdge(self):
        (i, j) = np.random.randint(0, self.vertexNumber, 2)
        if i in [self.inputNumber, self.inputNumber + self.outputNumber]:
            return 0
        if i in self.vertex[j].voisins:
            indice = self.vertexNumber
            E = Vertex(indice, leakyReLU if np.random.random() < 0.5 else Sigmoid, vertexType = 'inner')
            E.voisins = [i]
            E.dataOut = 0
            E.weights = [np.random.normal()]
            self.vertex.append(E)
            self.vertexNumber += 1
            self.vertex[j].voisins = [indice if x == i else x for x in self.vertex[j].voisins]
        else:
            indice = self.vertexNumber
            E = Vertex(indice, leakyReLU if np.random.random() < 0.5 else Sigmoid, vertexType = 'inner')
            E.voisins = [i]
            E.dataOut = 0
            E.weights = [np.random.normal()]
            self.vertex.append(E)
            self.vertexNumber += 1
            self.vertex[j].voisins.append(indice)
            self.vertex[j].weights.append(np.random.normal())

    def evaluate(self, dataloader):
        somme = 0
        sommepartielle = 0
        for _ in range(ITERATIONS):
            for i in range(dataloader.longueur):
                inD, outD = dataloader[np.random.randint(0, dataloader.longueur)]
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

    def renormalize(self):
        for N in self.population:
            N.initRandom(fonctionRandom, fonctionRandom, fonctionRandom)


DATAIN = [[0, 0], [1, 0], [0, 1], [1, 1]]
EXPECT = [[0], [1], [1], [0]]
dataloader = Dataloader(DATAIN, EXPECT)
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

P = Pool(250, dataloader, 3, 2, 1, [leakyReLU, leakyReLU, leakyReLU])


def plotBest(i = 0):
    G = nx.DiGraph()
    P.evaluate()
    N = P.population[i]
    number = N.vertexNumber
    for i in range(number):
        G.add_node(getLetter(i))
    for i in range(number):
        voisins = N.vertex[i].voisins
        weights = N.vertex[i].weights
        for j in range(len(voisins)):
            G.add_edge(getLetter(voisins[j]), getLetter(i), weights = weights[j])
    #Dessin:
    nx.draw(G, with_labels = True)
    plt.show()
