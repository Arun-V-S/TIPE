import numpy as np
import random
import matplotlib.pyplot as plt
import copy

plt.ion()

class Fonction:
    def __init__(self, ecartType):
        self.data = list(np.random.normal(scale = ecartType, size = 1))
        self.erreur = 0

    def evaluate(self, data):
        res = 0
        for i in range(len(self.data)):
            res += self.data[i] * (data ** i)
        return res

    def evaluateAll(self, datain):
        res = []
        for i in range(len(datain)):
            res.append(self.evaluate(datain[i]))
        return res

    def test(self, data, expect):
        erreur = 0
        for i in range(len(expect)):
            erreur += (expect[i] - self.evaluate(data[i])) ** 2
        self.erreur = erreur / len(expect)


class Batch:
    def __init__(self, number, ecartType):
        self.fonctions = []
        self.fonctionsNumber = number
        self.ecartType = ecartType
        for _ in range(number):
            self.fonctions.append(Fonction(ecartType))

    def testAll(self, data, expect):
        for F in self.fonctions:
            F.test(data, expect)

    def sortFonctions(self):
        self.fonctions = sorted(self.fonctions, key=lambda objet : objet.erreur)

    def evolve(self, keep, mutationRate, mutationEffect, exposantProbability):
        fonctions = self.fonctions[:keep]
        for _ in range(keep, self.fonctionsNumber):
            fonctions.append(copy.deepcopy(self.fonctions[np.random.randint(0, keep)]))

        for i in range(keep, self.fonctionsNumber):
            fonction = fonctions[i]
            for j in range(0, len(fonction.data)):
                if random.random() <= mutationRate:
                    fonction.data[j] += random.gauss(0, self.ecartType) * mutationEffect
            if random.random() <= exposantProbability:
                fonction.data.append(random.gauss(0, self.ecartType))
        self.fonctions = fonctions

    def evolveEpochs(self, epochs, keep, mutationRate, mutationEffect, exposantProbability, datain, expect):
        for i in range(epochs):
            self.testAll(datain, expect)
            self.sortFonctions()
            if i % 50 == 0:
                print(i)
                self.plotBest(datain, expect)
                print(self.bestFunction())
                print(self.fonctions[0].data)
            self.evolve(keep, mutationRate, mutationEffect, exposantProbability)

    def bestFunction(self):
        return self.fonctions[0].erreur

    def plotBest(self, datain, expect):
        res = self.fonctions[0].evaluateAll(datain)
        plt.clf()
        plt.ylim([min(expect) - 1, max(expect) + 1])
        plt.xlim([min(datain) - 1, max(datain) + 1])
        plt.plot(datain, expect, color = 'k')
        plt.plot(datain, res, color = 'b')
        plt.show()
        plt.pause(0.01)

DATAIN = [i for i in range(5)]
EXPECT = list(np.random.randint(0, 2, 5))

B = Batch(1000, 0.1)
B.evolveEpochs(2500, 500, 0.5, 1, 0.25, DATAIN, EXPECT)
