import numpy as np

def save(filename, sfile):
    np.save(filename, sfile)

class Network:
    """Simple neural network without bias for the moment"""
    def __init__(self, sizes):
        self.num_layers = len(sizes)

        self.layers = []
        for s1, s2 in zip(sizes[1:], sizes[:-1]):
            self.layers.append(2 * np.random.random((s2, s1)) - 1)
        
        self.error = 1
        
    def feedForward(self, a):
        """Retourne le résultat du réseau de neurones quand "a" est l'entrée"""
        for layer in self.layers:
            a = self.__sigmoid(np.dot(a, layer))
        return a

    def train(self, input_data, epochs, eta):
        for j in range(epochs):
            np.random.shuffle(input_data)
            self.backPropagation(input_data[:,:-1], np.array([input_data[:,-1]]).T, eta)

            if j % 10 == 0:
                print(f'Error : {np.mean(np.abs(self.error))}')
        
        save("layer1.npy", self.layers[0])
        save("layer2.npy", self.layers[1])
        save("layer3.npy", self.layers[2])
        save("layer4.npy", self.layers[3])

    def backPropagation(self, training_data, training_output, eta):
        # feedforward
        activation = training_data
        activations = [activation]
        zs = []

        for layer in self.layers:
            z = np.dot(activation, layer)
            zs.append(z)
            activation = self.__sigmoid(z)
            activations.append(activation)

        # backwardpass
        deltas = []
        self.error = activations[-1] - training_output

        delta = self.error * self.__sigmoidPrime(zs[-1])
        deltas.append(delta)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.__sigmoidPrime(z)
            delta = np.dot(delta, self.layers[-l+1].T) * sp
            deltas.append(delta)
        
        new_layers = []
        for layer, delta, a in zip(self.layers, deltas[::-1], activations):
            new_layers.append(layer - eta * np.dot(a.T, delta))
        self.layers = new_layers

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __sigmoidPrime(self, z):
        return self.__sigmoid(z) * (1 - self.__sigmoid(z))