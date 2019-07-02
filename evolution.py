import numpy as np
import random

class Fonction:
    def __init__(self, ecartType):
        self.data = list(np.random.normal(scale = ecartType, size = 1))
        self.erreur = 0

    def evaluate(self, data):
        res = 0
        for i in range(len(self.data)):
            res += self.data[i] * (data ** i)
        return res

    def test(self, data, expect):
        erreur = 0
        for i in range(len(expect)):
            erreur += (expect[i] - self.evaluate(data[i])) ** 2
        self.erreur = erreur


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

    def evolve(self, keep, mutationRate, mutationEffect, exposantProbability):
        self.fonctions = sorted(self.fonctions, key=lambda objet : objet.erreur)
        fonctions = self.fonctions[:keep]
        for _ in range(keep, self.fonctionsNumber):
            fonctions.append(Fonction(self.ecartType))
        self.fonctions = fonctions

        for i in range(0, keep):
            fonction = self.fonctions[i]
            for j in range(len(fonction.data)):
                if random.random() <= mutationRate:
                    fonction.data[j] += random.gauss(0, self.ecartType) * mutationEffect
            if random.random() <= exposantProbability:
                fonction.data.append(random.gauss(0, self.ecartType))
        self.fonctions = fonctions

    def bestFunction(self):
        return self.fonctions[0].erreur

    def train(self, datain, expect, epochsNumber, keep, mutationRate, mutationEffect, exposantProbability):
        for i in range(epochsNumber):
            B.testAll(datain, expect)
            B.evolve(keep, mutationRate, mutationEffect, exposantProbability)
            print(i)
            print(self.bestFunction())

DATAIN = [0, 1, 2, 3, 4, 5, 6]
EXPECT = [-4, 6, 12, 4, 3, 2, -2]

B = Batch(100, 1)
B.testAll([1], [1])
B.evolve(4, 4, 4, 4)
