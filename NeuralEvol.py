## Algorithme d'évolution de réseaux de neurones
## Ces réseaux sont représentés sous la forme d'un graphe orienté

class Edge:
    def __init__(self, number, edgeType = 'inner'):
        self.number = number
        self.dataIn = None #Besoin de différencier les deux pour le cas de récurrences,
        self.dataOut = None #permettant de bien faire avancer le réseau par paliers
        self.weights = []
        self.voisins = [] # Il s'agit des voisins en amont!
        self.bias = 0
        self.edgeType = edgeType

    def __repr__(self):
        return str(self.number)

    def feedDataIntern(self):
        self.dataOut = self.dataIn

class Network:
    def __init__(self, edgeNumber, inputNumber, outputNumber):
        assert inputNumber + outputNumber <= edgeNumber, 'probleme de nombre de neurones'
        self.edgeNumber = edgeNumber
        self.inputNumber = inputNumber
        self.outputNumber = outputNumber
        self.inputEdges = [i for i in range(inputNumber)]
        self.outputEdges = [i for i in range(inputNumber, inputNumber + outputNumber)]
        self.edges = []
        for i in range(inputNumber):
            self.edges.append(Edge(i, 'input'))
        for i in range(inputNumber, inputNumber + outputNumber):
            self.edges.append(Edge(i, 'output'))
        for i in range(inputNumber + outputNumber, edgeNumber):
            self.edges.append(Edge(i))
