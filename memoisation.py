import copy
import inspect

class Memoise():
    def __init__(self, fonction):
        self.fonction = fonction
        self.valeurs = {}

    def verif(self, *args):
        res = ""
        for arg in args:
            attributes = return_attributes(arg)
            for at in attributes:
                key, value = at
                res += str(value)
                res += '&'
        if res in self.valeurs.keys():
            return self.valeurs[res]
        else:
            retour = self.fonction(*args)
            self.valeurs[res] = retour
            return retour

    def __call__(self, *args):
        return self.verif(*args)

    def clear(self):
        del self.valeurs
        self.valeurs = {}


def memoisation(fonction):
    return Memoise(fonction)


class Slime:
    ab = 5
    def __init__(self):
        self.at = 5

def affiche(slime):
    print("Affichage :")
    print(slime.at)

affichage = memoisation(affiche)


def return_attributes(objet):
    listeTotale = inspect.getmembers(objet, lambda a : not(inspect.isroutine(a)))
    res = []
    for e in listeTotale:
        if e[0][0] != '_':
            res.append(e)
    return res


def fiboRec(nombre):
    if nombre >= 2:
        return fiboRec(nombre - 1) + fiboRec(nombre - 2)
    else:
        return 1

fiboRecMem = memoisation(fiboRec)

class ADN:
    def __init__(self, **chr):
        self.__dict__.update(chr)
        self.mutationRate = 5

    def addAttr(self, **kwds):
        self.__dict__.update(kwds)

    def mutate(self, rate = None):
        if rate == None:
            rate = self.mutationRate

class Chromosome:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def mutate(self, rate):
        pass
