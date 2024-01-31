from random import random
from math import ceil, exp

class NeuralNetwork:
    def __init__(self, layers_cfg, input_nodes=1, learning_rate=0.1):
        self.input_nodes = input_nodes
        self.learning_rate = learning_rate
        self.layers = []

        for i in range(len(layers_cfg)):
            cols = layers_cfg[i - 1]['nodes']
            if i == 0:
                cols = input_nodes
            layer = Layer(
                layers_cfg[i]['nodes'],
                layers_cfg[i]['activation_func'],
                Matrix(layers_cfg[i]['nodes'], cols),
                Matrix(layers_cfg[i]['nodes'], 1)
            )
            layer.weights.random(-1, 1)
            layer.bias.random(-1, 1)
            self.layers.append(layer)

    def generate_outputs(self, inputs):
        prev_layer_values = Matrix.from_array(inputs)
        for layer in self.layers:
            layer.values = layer.weights.multiply(prev_layer_values)
            layer.values = layer.values.add(layer.bias)
            layer.values = layer.values.map(layer.afunc)
            prev_layer_values = layer.values
        outputs = prev_layer_values
        return outputs
    
    def predict(self, inputs):
        return self.generate_outputs(inputs).to_array()
    
    def train(self, input_array, target_array):
        outputs = self.generate_outputs(input_array)
        targets = Matrix.from_array(target_array)

        prev_error = None

        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            if prev_error is None:
                error = targets.substract(outputs)
            else:
                next_weights = self.layers[i + 1].weights
                next_weights_t = next_weights.transpose()
                error = next_weights_t.multiply(prev_error)
            gradient = layer.values.map(layer.dafunc)
            gradient = gradient.multiply(error)
            gradient = gradient.multiply_n(self.learning_rate)

            inputs = Matrix.from_array(input_array)
            prev_values = inputs if i == 0 else self.layers[i - 1].values
            prev_values_t = prev_values.transpose()
            layer_weight_deltas = gradient.multiply(prev_values_t)

            layer.weights = layer.weights.add(layer_weight_deltas)
            layer.bias = layer.bias.add(gradient)

            prev_error = error 


def sigmoid(x):
    if x < 0:
        return 1 - 1 / (1 + exp(x))
    return 1 / (1 + exp(-x))

def dsigmoid(y):
    return y * (1 - y)

class Layer:
    def __init__(self, nodes, activation_func, weights, bias):
        self.nodes = nodes
        self.weights = weights
        self.bias = bias
        self.values = None

        if activation_func == 'sigmoid':
            self.afunc = sigmoid
            self.dafunc = dsigmoid

class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        self.matrix = []

        for i in range(rows):
            self.matrix.append([])
            for j in range(cols):
                self.matrix[i].append(0)
        
    def random(self, min, max):
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] = random() * (max - min) + min

    def __str__(self):
        s = ''
        for i in range(self.rows):
            s += str(self.matrix[i]) + '\n'
        return s
    
    def map(self, f):
        nm = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                v = self.matrix[i][j]
                nm.matrix[i][j] = f(v)
        return nm

    def add(self, m):
        if self.rows == m.rows and self.cols == m.cols:
            nm = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    nm.matrix[i][j] = self.matrix[i][j] + m.matrix[i][j]
            return nm

    def substract(self, m):
        if self.rows == m.rows and self.cols == m.cols:
            nm = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    nm.matrix[i][j] = self.matrix[i][j] - m.matrix[i][j]
            return nm

    def multiply(self, m):
        if self.rows == m.rows and self.cols == m.cols:
            nm = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    nm.matrix[i][j] = self.matrix[i][j] * m.matrix[i][j]
            return nm        
        elif self.cols == m.rows:
            nm = Matrix(self.rows, m.cols)
            for i in range(nm.rows):
                for j in range(nm.cols):
                    sum = 0
                    for k in range(self.cols):
                        sum += self.matrix[i][k] * m.matrix[k][j]
                    nm.matrix[i][j] = sum
            return nm

    def multiply_n(self, n):
        nm = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                nm.matrix[i][j] = self.matrix[i][j] * n
        return nm       
        
    def transpose(self):
        nm = Matrix(self.cols, self.rows)
        for i in range(nm.rows):
            for j in range(nm.cols):
                nm.matrix[i][j] = self.matrix[j][i]
        return nm

    @staticmethod
    def from_array(array, cols=1):
        rows = ceil(len(array) / cols)
        nm = Matrix(rows, cols)
        for i in range(rows):
            for j in range(cols):
                index = i * cols + j
                nm.matrix[i][j] = array[index]
        return nm

    def to_array(self):
        array = []
        for i in range(self.rows):
            for j in range(self.cols):
                array.append(self.matrix[i][j])
        return array
    
    def copy(self):
        c = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                c.matrix[i][j] = self.matrix[i][j]
        return c
                