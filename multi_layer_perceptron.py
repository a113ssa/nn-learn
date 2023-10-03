import numpy as np
from perceptron import Perceptron


class MultiLayerPerceptron:
    def __init__(self, layers, bias=1.0):
        self.layers = np.array(layers, dtype=object)
        self.bias = bias
        self.network = []
        self.values = []

        for i in range(len(self.layers)):
            self.values.append([])
            self.network.append([])
            self.values[i] = [0.0 for _ in range(self.layers[i])]

            if i > 0:
                for _ in range(self.layers[i]):
                    perceptron = Perceptron(
                        inputs=self.layers[i - 1], bias=self.bias)
                    self.network[i].append(perceptron)

        self.network = self.convert_to_numpy_array(self.network)
        self.values = self.convert_to_numpy_array(self.values)

    def convert_to_numpy_array(self, array):
        return np.array([np.array(x, dtype=object) for x in array], dtype=object)

    def set_weights(self, weights):
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                self.network[i + 1][j].set_weights(weights[i][j])

    def print_weights(self):
        print('Weights:')
        for i in range(1, len(self.network)):
            for j in range(len(self.network[i])):
                print("Layer: ", (i + 1), "Neuron: ",
                      j, self.network[i][j].weights)

    def run(self, x):
        x = np.array(x, dtype=object)
        self.values[0] = x
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                self.values[i][j] = self.network[i][j].run(self.values[i - 1])
        return self.values[-1]


if __name__ == '__main__':
    mlp = MultiLayerPerceptron(layers=[2, 2, 1])
    mlp.set_weights([[[-10, -10, 15], [10, 10, -5]], [[10, 10, -15]]])
    mlp.print_weights()

    def print_result(mlp):
        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        for i in inputs:
            print("Input: ", i, "Output: ", mlp.run(i)[0])

    print("MLP:")
    print_result(mlp)
