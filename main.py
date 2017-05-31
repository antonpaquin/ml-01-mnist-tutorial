import numpy as np
import random
from copy import deepcopy


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        self.delta_bias_momentum = [np.zeros(b.shape) for b in self.biases]
        self.delta_weight_momentum = [np.zeros(w.shape) for w in self.weights]

    def feedforward(self, x):
        for bias, weight in zip(self.biases, self.weights):
            x = sigmoid(np.dot(weight, x) + bias)
        return x

    def gradient_descent(self, training_data, epochs, batch_size, learning_rate, momentum, test_data=None):
        if test_data:
            n_test = len(test_data)

        n_train = len(training_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0, n_train, batch_size)]
            for batch in batches:
                self.update_batch(batch, learning_rate, momentum)
            if test_data:
                score = self.evaluate(test_data)
                print("Epoch {epoch}: {success} / {max}".format(
                    epoch=epoch, success=score, max=n_test
                ))
            else:
                print("Epoch {epoch} done".format(epoch=epoch))

    def backprop(self, inp, out):

        grad_bias = [np.zeros(b.shape) for b in self.biases]
        grad_weight = [np.zeros(w.shape) for w in self.weights]

        activation = inp
        activations = [deepcopy(activation)]
        for weight, bias in zip(self.weights, self.biases):
            activation = sigmoid(np.dot(weight, activation) + bias)
            activations.append(deepcopy(activation))

        err = costf_p(out, activation)
        for l in range(-1, -self.num_layers, -1):
            grad_bias[l] = deepcopy(err)
            grad_weight[l] = np.dot(err, activations[l-1].transpose())
            err = np.dot(self.weights[l].transpose(), err)

        return grad_bias, grad_weight

    def update_batch(self, batch, learning_rate, momentum):
        delta_bias = [np.zeros(b.shape) for b in self.biases]
        delta_weight = [np.zeros(w.shape) for w in self.weights]

        for inp, out in batch:
            adj_bias, adj_weights = self.backprop(inp, out)
            delta_bias = [delta + adj for delta, adj in zip(delta_bias, adj_bias)]
            delta_weight = [delta + adj for delta, adj in zip(delta_weight, adj_weights)]

        for l in range(self.num_layers - 1):
            delta_bias[l] = momentum * delta_bias[l] + (1-momentum) * self.delta_bias_momentum[l]
            delta_weight[l] = momentum * delta_weight[l] + (1-momentum) * self.delta_weight_momentum[l]

            self.delta_bias_momentum[l] = delta_bias[l]
            self.delta_weight_momentum[l] = delta_weight[l]

        self.biases = [bias - (learning_rate * (delta / len(batch)))
                       for bias, delta in zip(self.biases, delta_bias)]
        self.weights = [weight - (learning_rate * (delta / len(batch)))
                        for weight, delta in zip(self.weights, delta_weight)]

    def evaluate(self, test_data):
        classification = [(np.argmax(self.feedforward(inp)), out) for (inp, out) in test_data]
        return sum([int(actual == expected) for actual, expected in classification])


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sig_prime(x):
    return x * (1-x)


def costf_p(expect, actual):
    return actual - expect
