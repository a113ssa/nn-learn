import numpy as np


class Perceptron:
    def __init__(self, inputs, bias=1):
        self.weights = (np.random.rand(inputs + 1)) * 2 - 1
        self.bias = bias

    def run(self, inputs):
        x_sum = np.dot(np.append(inputs, self.bias), self.weights)
        return self.sigmoid(x_sum)

    def set_weights(self, w_init):
        self.weights = np.array(w_init)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    def print_result(neuron):
        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        for i in inputs:
            print("Input: ", i, "Output: ", neuron.run(i))

    neuron = Perceptron(2)
    neuron.set_weights([10, 10, -15])
    print("AND Gate")
    print_result(neuron)

    neuron.set_weights([10, 10, -5])
    print("OR Gate")
    print_result(neuron)

    neuron.set_weights([-10, -10, 15])
    print("NAND Gate")
    print_result(neuron)

    neuron.set_weights([-10, -10, 5])
    print("NOR Gate")
    print_result(neuron)
