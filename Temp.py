class Layer:
    def __init__(self, name, tin, tout, type = 'hidden'):
        """initialise la couche."""

    def propagate(self, target):
        """Fait la propagation de self à target."""

    def putdata(self, data):
        """Initialise la data de self à data."""

    def returndata(self):
        """Renvoie les données dans les neurones de la couche."""

class Network:
    def __init__(self, name, Layers, number):
        """initialise le réseau"""

    def save(self, name):
        """Sauvegarde le réseau sous forme de texte."""

    def load(self, name):
        """Charge un réseau enregistré"""

    def inputdata(self, data):
        """Rentre les données dans la couche input du réseau."""

    def outputdata(self):
        """Sort les données de la couche output du réseau."""

    def propagatelayers(self):
        """Fait la propagation de tout le réseau."""

    def test(self, data):
        """Renvoie le résultat du traitement par le réseau de data."""

    def loss(self, DATAIN, EXPECT):
        """Fait la fonction loss sur un paquet de données."""

    def train(self, DATAIN, EXPECT, pers, number):

    def backward(self, datain, expect, pers):
