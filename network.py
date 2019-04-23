import numpy as np

def save(filename, sfile):
    """Permet de sauvegarder des fichiers (ici pour sauvegarder les poids des différentes couches"""
    np.save(filename, sfile)

class Network:
    """Un réseau de neurones simple, sans biais pour le moment"""
    def __init__(self, sizes):
        self.num_layers = len(sizes)

        self.layers = []
        for s1, s2 in zip(sizes[1:], sizes[:-1]):
            self.layers.append(2 * np.random.random((s2, s1)) - 1)
        
        self.error = 1

    def load(self):
        """Permet d'initialiser les valeurs des poids des différentes couches pour un réseau qui a déjà été entrainé"""
        l1 = np.load("layer1.npy")
        l2 = np.load("layer2.npy")
        l3 = np.load("layer3.npy")
        
        self.layers = [l1, l2, l3]
        
    def feedForward(self, a):
        """Retourne le résultat du réseau de neurones quand "a" est l'entrée"""
        for layer in self.layers:
            # Calcul du vecteur en sortie de chaque couche
            a = self.__sigmoid(np.dot(a, layer))

        # Permet d'arrondir le vecteur de sortie (pour correspondre à notre problématique)
        out = np.zeros(a.shape)
        for i, val in enumerate(a):
            if val >= 0.5:
                out[i] = 1
            else:
                out[i] = 0
        return out

    def train(self, input_data, output_data, epochs, batch, eta):
        # On travail avec des petits groupes de données plutôt qu'avec un gros groupe de donnée (permet d'accélérer les calculs)
        a = 0
        b = batch
        for _ in range(0, len(input_data), batch):
            for j in range(epochs):
                self.backPropagation(input_data[a:b], output_data[a:b], eta)

                if j % 50 == 0:
                    print(f'Error : {np.sum(np.abs(self.error)) / batch}')

                    # Sauvegarder les valeurs des poids des différentes couches pour pouvoir réutiliser le réseau
                    save("layer1.npy", self.layers[0])
                    save("layer2.npy", self.layers[1])
                    save("layer3.npy", self.layers[2])

            a += batch
            b += batch
    
    def score(self, input_data, output_data):
        """Permet de connaitre la précision du réseau de neurones avec des données qu'ils n'a jamais vu"""
        act = self.feedForward(input_data)
        diff = np.abs(act - output_data)
        return 1 - np.sum(diff) / len(diff)

    def backPropagation(self, training_data, training_output, eta):
        """Algorithme de rétro-propagation"""

        # Propagation
        activation = training_data
        activations = [activation]
        zs = []

        for layer in self.layers:
            z = np.dot(activation, layer)
            zs.append(z)
            activation = self.__sigmoid(z)
            activations.append(activation)

        # Rétro-propagation
        deltas = []
        self.error = activations[-1] - training_output

        delta = self.error * self.__sigmoidPrime(zs[-1])
        deltas.append(delta)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.__sigmoidPrime(z)
            delta = np.dot(delta, self.layers[-l+1].T) * sp
            deltas.append(delta)
        
        # Correction des poids pour chaques couches
        new_layers = []
        for layer, delta, a in zip(self.layers, deltas[::-1], activations):
            new_layers.append(layer - eta * np.dot(a.T, delta))
        self.layers = new_layers

    def __sigmoid(self, z):
        """Fonction d'activation sigmoïd"""
        return 1 / (1 + np.exp(-z))
    
    def __sigmoidPrime(self, z):
        """Dérivée de la fonction sigmoïd"""
        return self.__sigmoid(z) * (1 - self.__sigmoid(z))