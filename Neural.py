import numpy as np
import math
import copy
import matplotlib.pyplot as plt

def sigmoid(x):
    try:
        ans = 1 / (1 + np.exp(-x))
    except OverflowError:
        ans = 0.5
        print('Overflow                   !!!!!!!')
    return ans

def sigmoidder(x):
    return sigmoid(x) * (1 - sigmoid(x))

def logit(x): #Réciproque de la sigmoïde
    return np.ln(x / (1 - x))

class Function:
    def __init__(self, fonc, der): #On définit une fonction à partir de son expression et sa dérivée
        self.fonc = fonc
        self.der = der

class Layer:
    def __init__(self, name, tin, tout, type = 'hidden'): #tin et tout sont des tailles, type peut être input, output ou hidden
        self.name = name
        self.type = type
        self.sizein = tin
        self.sizeout = tout
        #if type != 'output':
        self.weights = np.random.rand(self.sizein, self.sizeout) * 2 - 1
        #self.weights = np.full((self.sizein, self.sizeout), 0.5)
        self.biases = np.random.rand(self.sizein) #* 2 - 1
        #self.biases = np.zeros(self.sizein)
        self.data = np.zeros(self.sizein)

    def propagate(self, target):
        """Fait la propagation de self à target."""
        if self.sizeout != target.sizein:
            print('Erreur de dimension ' + self.name)
            return 0
        res = sigmoid(np.dot(self.data, self.weights) + target.biases)
        target.data = res

    def putdata(self, data):
        """Initialise la data de self à data."""
        if len(data) != self.sizein:
            print('Erreur de dimension ' + self.name)
            return 0
        for i in range(len(data)):
            self.data[i] = data[i]

    def printdata(self):
        print(self.name + ' data')
        for x in self.data:
            print(x)

    def printweights(self):
        print(self.name + ' weights')
        for x in self.weights:
            print(x)

    def printbiases(self):
        print(self.name + ' biases')
        for x in self.biases:
            print(x)

    def returndata(self):
        if self.type != 'output':
            print("Attention, ce n'est pas une couche output _returndata_ " + self.name)
        return self.data


class Network:
    def __init__(self, name, Layers, number):
        self.layers = Layers
        self.name = name
        self.layersnumber = number
        self.epochs = 0 #Le nombre de cycles de retropropagation deja faits

    def save(self, name):
        """Sauvegarde le réseau sous forme de texte."""
        Texte = name + '!'
        for i in range(self.layersnumber):
            couche = self.layers[i]
            Texte += couche.name + ';' + str(couche.sizein) + ';' + str(couche.sizeout) + ';' + str(list(couche.weights)) + ';' + str(list(couche.biases)) + ';' + couche.type
            Texte += '!'
        with open(name + '.txt', 'w') as fichier:
            fichier.write(Texte)

    def load(self, name):
        LAYERS = []
        self.layers = []
        self.layersnumber = 0
        if not ('.txt' in name):
            name += '.txt'
        with open(name, 'r') as fichier:
            data = fichier.readlines()[0].strip()
            tout = data.split('!')
            self.name = tout[0]
            couches = tout[1:]
        for i in range(len(couches)):
            elements = couches[i].split(';')
            print(elements)
            LAYERS.append(Layer(elements[0], int(elements[1]), int(elements[2]), np.array(elements[3]), np.array(elements[4]), type = elements[5]))
        self.layers = LAYERS
        self.layersnumber = len(couches)

    def inputdata(self, data):
        """Rentre les données dans la couche input du réseau."""
        self.layers[0].putdata(data)

    def outputdata(self):
        """Sort les données de la couche output du réseau."""
        return (self.layers[self.layersnumber - 1]).returndata()

    def propagatelayers(self):
        """Fait la propagation de tout le réseau."""
        for i in range(self.layersnumber - 1):
            self.layers[i].propagate(self.layers[i + 1])

    def test(self, data):
        self.inputdata(data)
        self.propagatelayers()
        return self.outputdata()

    def lossOne(self, datain, expect):
        """Propage le réseau avec datain en entrée et renvoie [(expect - real) ** 2]."""
        self.inputdata(datain)
        self.propagatelayers()
        resultat = self.outputdata()
        lenExpect = len(expect)
        ret = np.zeros(lenExpect)
        if lenExpect != len(resultat):
            print('Erreur de dimension _test_')
            return 0
        res = 0
        for i in range(lenExpect):
            res += (expect[i] - resultat[i]) ** 2
        return res

    def lossAll(self, DATAIN, EXPECT):
        """Fait la fonction loss sur un paquet de données."""
        loss = 0
        for i in range(len(DATAIN)):
            datain = DATAIN[i]
            expect = EXPECT[i]
            loss += self.lossOne(datain, expect)
        return loss

    def train(self, DATAIN, EXPECT, pers, number):
        global plot
        n = len(DATAIN)
        for i in range(number):
            print(i)
            perte = self.lossAll(DATAIN, EXPECT)
            if plot:
                LOSS.append(perte)
            print(perte)
            for j in range(n):
                #persistance = pers * np.exp(- i)
                self.backward(DATAIN[j], EXPECT[j], pers)
        self.epochs += number

    def backward(self, datain, expect, pers):
        self.inputdata(datain)
        self.propagatelayers()
        result = self.outputdata()
        DELTA = []
        for i in reversed(range(0, self.layersnumber)):
            if i == self.layersnumber - 1:
                data = self.layers[i].data
                """delta = (data - expect) * data * (1 - data)"""
                delta = (np.expand_dims(data, 0) - np.expand_dims(expect, 0)) * np.expand_dims(data, 0) * (1 - np.expand_dims(data, 0))
                deltai = copy.deepcopy(delta[0,:])
                DELTA.append(deltai)
            else:
                data = self.layers[i].data
                res = np.zeros(self.layers[i].sizein)
                for a in range(self.layers[i].sizein):
                    for j in range(self.layers[i].sizeout):
                        #print(deltai[j] * self.layers[i].weights[a, j])
                        res[a] += deltai[j] * self.layers[i].weights[a, j]
                issou = (res * np.expand_dims(data, 0) * (1 - np.expand_dims(data, 0)))[0,:]
                DELTA.append(issou)
                deltai = copy.deepcopy(issou)
                #print(issou)
                #print(DELTA)
        for i in range(len(DELTA) - 1):
            data = np.expand_dims(copy.deepcopy(self.layers[self.layersnumber - i - 2].data), axis=0)
            delta = np.expand_dims(copy.deepcopy(DELTA[i]), axis = 0)
            res = pers * data.T.dot(delta)
            self.layers[self.layersnumber - i - 2].weights -= res

#DATAIN = np.array([np.array([1., 0.]), np.array([0., 1.])])
#EXPECT = np.array([np.array([1., 0.]), np.array([0., 1.])])
#DATAIN = np.array([np.array([np.random.rand(), np.random.rand()]) for _ in range(100)])
DATAIN = np.concatenate((np.random.rand(250, 1), - np.random.rand(250, 1)))
EXPECT = np.concatenate( (np.full(250, [1, 0], dtype='2f'), np.full(250, [0, 1], dtype='2f')) )
#DATAIN = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
#EXPECT = np.array([[0, 0], [1, 0], [1, 0], [0, 1]])



"""for i in range(100):
    DATAIN[0, i], DATAIN[1, i] = np.random.rand(), np.random.rand()"""

Layer1 = Layer('1', 1, 2, type='input')
Layer2 = Layer('2', 2, 4, type='hidden')
#Layer3 = Layer('3', 4, 4, type='hidden')
Layer4 = Layer('4', 4, 2, type='hidden')
Layer5 = Layer('5', 2, 1, type='output')

Reseau = Network('Test', [Layer1, Layer2, Layer4, Layer5], 4)
plot = True
if plot:
    LOSS = []
    def plotloss():
        time = np.arange(0, len(LOSS))
        plt.plot(time, LOSS, label='Coût selon le nombre de générations')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.xlabel('Générations')
        plt.ylabel('Coût')
        plt.savefig('Cost')
        plt.show()

TEST = np.arange(-10, 10, 0.1)

def plotPosNeg(name, title):
    POSITIF = []
    NEGATIF = []
    for i in range(len(TEST)):
        pos, neg = Reseau.test([TEST[i]])
        POSITIF.append(pos)
        NEGATIF.append(neg)
    plt.plot(TEST, POSITIF, label = 'Sortie -')
    plt.plot(TEST, NEGATIF, label = 'Sortie +')
    plt.xlabel('Nombre en entrée')
    plt.ylabel('Résultats')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    plt.savefig(name + '.png')

#Reseau.train([DATAIN[0]], [DATAIN[0]], 0.1, 100)
#Reseau.train([[1, 0]], [[1, 0]], 0.1, 10)
"""Reseau.train(DATAIN[0], EXPECT[0], 0.01, 50)
Reseau.inputdata(DATAIN[0])
Reseau.propagatelayers()
print(Reseau.outputdata())"""
