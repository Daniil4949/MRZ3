# Лабораторная работа 3 по дисциплине МРЗвИС
# Выполнена студентом группы 121702 БГУИР Кимстач Д.Б.
# Реализация сети Джордана
# Вариант 11
# Ссылки на источники:
# https://rep.bstu.by/bitstream/handle/data/30365/471-1999.pdf?sequence=1&isAllowed=y

import numpy as np


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def dleaky_relu(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


def step(n):
    return n ** 2


class JordanNetwork:
    def __init__(self, *args, scale_k=100, activation=leaky_relu, d_activation=dleaky_relu):
        self.scale_k = scale_k
        self.shape = args
        self.activation = activation
        self.d_activation = d_activation
        self.layers = [np.ones(self.shape[0] + self.shape[-1])]
        self.layers.extend(np.ones(shape) for shape in self.shape[1:])
        self.context = np.zeros(self.shape[-1])
        self.weights = [
            np.random.uniform(-1, 1, (self.layers[i].size, self.layers[i + 1].size))
            for i in range(len(self.shape) - 1)
        ]
        self.dw = [np.zeros_like(weight) for weight in self.weights]

    def propagate_forward(self, data):
        data = data / self.scale_k
        self.layers[0][:self.shape[0]] = data
        self.layers[0][self.shape[0]:] = self.context
        for i in range(1, len(self.shape)):
            self.layers[i] = self.activation(np.dot(self.layers[i - 1], self.weights[i - 1]))
        self.context = self.layers[-1]
        return self.layers[-1] * self.scale_k

    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        target = target / self.scale_k
        deltas = []
        error = target - self.layers[-1]
        delta = error * self.d_activation(self.layers[-1])
        deltas.append(delta)
        for i in range(len(self.shape) - 2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.d_activation(self.layers[i])
            deltas.insert(0, delta)
        for i, (layer, delta) in enumerate(zip(self.layers[:-1], deltas)):
            dw = np.dot(np.atleast_2d(layer).T, np.atleast_2d(delta))
            self.weights[i] += lrate * dw + momentum * self.dw[i]
            self.dw[i] = dw
        return (error ** 2).sum()


def generate_train_matrix(sequence_function, rows, cols):
    return np.array([[sequence_function(j + i) for j in range(cols)] for i in range(rows)])


def generate_train_matrix_result(sequence_function, rows, offset, cols):
    return np.array([[sequence_function(j + i + offset) for j in range(cols)] for i in range(rows)])
