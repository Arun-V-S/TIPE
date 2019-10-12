import numpy as np
import matplotlib.pyplot as plt
import usuelles
import random

BREEDRATE = 0.25

class Food:
    def __init__(self):
        self.amount = 10

    def __repr__(self):
        return "\033[1;33;40mF " + str(self.amount) + "\033[1;37;40m"

class Slime:
    def __init__(self):
        self.vitesse = 0
        self.taille = 1
        self.foodMax = 40
        self.food = self.foodMax // 2
        self.turn = 0
        self.generation = 0


    def __repr__(self):
        return "\033[1;32;40mS " + str(self.generation) + "\033[1;37;40m"

    def eat(self, target, amount):
        target.amount -= amount
        self.food += amount
        self.food = min(self.foodMax, self.food)

    def fatigue(self, amount):
        self.food -= amount

    def getBreededGenes(self):
        return [max(1, self.vitesse + np.random.randint(-5, 6)), max(1, self.taille + np.random.randint(-1, 2)),
                max(1, self.foodMax + np.random.randint(-5, 6))]


class Terrain:
    def __init__(self, taille):
        self.taille = taille
        self.plateau = np.empty((taille, taille), dtype = object)

    def clear(self):
        self.plateau = np.empty((self.taille, self.taille), dtype = object)

    def spawnFood(self, rate, foodLimit):
        l = 0
        for i in range(self.taille):
            for j in range(self.taille):
                if random.random() <= rate and self.plateau[i, j] == None and l < foodLimit:
                    self.plateau[i, j] = Food()
                    l += 1

    def clearFood(self):
        for i in range(self.taille):
            for j in range(self.taille):
                if type(self.plateau[i, j]) == Food:
                    self.plateau[i, j] = None

    def spawnSlimes(self, number):
        if self.taille ** 2 < number:
            pass
        a = 0
        while a < number:
            i, j = tuple(np.random.randint(0, self.taille, 2))
            if self.plateau[i, j] == None:
                self.plateau[i, j] = Slime()
                a += 1

    def step(self, turn):
        for i in range(self.taille):
            for j in range(self.taille):
                if type(self.plateau[i, j]) == Slime and self.plateau[i, j].turn < turn:
                    S = self.plateau[i, j]
                    f = self.searchFood((i, j))
                    p = self.searchPlace((i, j))
                    if S.food >= S.foodMax // 2:
                        self.breed(S, (i, j))
                    if f != None and S.food < S.foodMax:
                        F = self.plateau[f[0], f[1]]
                        S.eat(F, min(S.taille, F.amount))
                        if F.amount == 0:
                            self.plateau[f[0], f[1]] = None
                    elif p != None:
                        self.plateau[p[0], p[1]] = S
                        self.plateau[i, j] = None
                        S.fatigue(S.taille)
                        if S.food <= 0:
                            self.plateau[p[0], p[1]] = None
                    S.turn = turn


    def searchFood(self, coords):
        possible = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                a, b = (usuelles.clipcoord(i + coords[0], self.taille), usuelles.clipcoord(j + coords[1], self.taille))
                if type(self.plateau[a, b]) == Food and (i, j) != (0, 0):
                    possible.append((a, b))
        if possible != []:
            return possible[np.random.randint(0, len(possible))]
        else:
            return None

    def searchPlace(self, coords):
        possible = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                a, b = (usuelles.clipcoord(i + coords[0], self.taille), usuelles.clipcoord(j + coords[1], self.taille))
                if self.plateau[a, b] == None and (i, j) != (0, 0):
                    possible.append((a, b))
        if possible != []:
            return possible[np.random.randint(0, len(possible))]
        else:
            return None

    def cycle(self, number, foodRate, foodLimit):
        for i in range(number):
            T.spawnFood(foodRate, foodLimit)
            self.step(i)
            #print(T.plateau)
            #T.clearFood()
            if i % 50 == 0:
                print(i)

    def breed(self, slime, coords):
        p = self.searchPlace(coords)
        if p != None and random.random() <= BREEDRATE:
            x, y = p
            genes = slime.getBreededGenes()
            S = Slime()
            S.vitesse = genes[0]
            S.taille = genes[1]
            S.foodMax = genes[2]
            S.food = S.foodMax // 2
            S.generation = slime.generation + 1
            self.plateau[p[0], p[1]] = S
            self.turn = slime.turn + 1
            slime.food //= 2

    def getTrucs(self):
        vitesse = []
        taille = []
        foodMax = []
        for i in range(self.taille):
            for j in range(self.taille):
                if type(self.plateau[i, j]) == Slime:
                    S = self.plateau[i, j]
                    vitesse.append(S.vitesse)
                    taille.append(S.taille)
                    foodMax.append(S.foodMax)
        plt.subplot(2, 2, 1)
        plt.hist(vitesse)
        plt.subplot(2, 2, 2)
        plt.hist(taille)
        plt.subplot(2, 2, 3)
        plt.hist(foodMax)
        plt.show()



T = Terrain(50)
T.spawnSlimes(250)
T.cycle(10000, 0.15, 10)
T.getTrucs()
