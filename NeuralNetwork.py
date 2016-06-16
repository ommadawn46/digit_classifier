# -*- coding:utf-8 -*-
import numpy as np


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1.0 - x ** 2


def softmax(x):
    t = np.exp(x)
    return t / np.sum(t)


class NeuralNetwork:
    def __init__(self, n1, n2, n3, act=tanh, d_act=d_tanh, act2=softmax):
        self.n1 = n1 + 1
        self.n2 = n2 + 1
        self.n3 = n3

        self.a1 = np.ones(self.n1)
        self.a2 = np.ones(self.n2)
        self.a3 = np.ones(self.n3)

        self.w1_2 = np.random.uniform(-1.0, 1.0, (self.n2, self.n1))
        self.w2_3 = np.random.uniform(-1.0, 1.0, (self.n3, self.n2))

        self.act = act
        self.d_act = d_act
        self.act2 = act2

    def update(self, inputs):
        self.a1[:-1] = inputs
        self.a2 = self.act(self.w1_2.dot(self.a1))
        self.a3 = self.act2(self.w2_3.dot(self.a2))

    def backprop(self, targets, n):
        d2_3 = self.a3 - np.array(targets)
        d1_2 = self.d_act(self.a2) * self.w2_3.T.dot(d2_3)

        a2 = np.atleast_2d(self.a2)
        d2_3 = np.atleast_2d(d2_3)
        self.w2_3 -= n * d2_3.T.dot(a2)

        a1 = np.atleast_2d(self.a1)
        d1_2 = np.atleast_2d(d1_2)
        self.w1_2 -= n * d1_2.T.dot(a1)

    def train(self, inputPatterns, targetPatterns, n=0.1):
        err = 0.0
        for inputs, targets in zip(inputPatterns, targetPatterns):
            self.update(inputs)
            self.backprop(targets, n)
            for k in range(len(targets)):
                err += 0.5 * (targets[k] - self.a3[k]) ** 2
        return err / len(inputPatterns)

    def test(self, inputPatterns, targetPatterns, print_output=False):
        outputs = []
        for inputs, targets in zip(inputPatterns, targetPatterns):
            self.update(inputs)
            outputs.append(self.a3[:])
            if print_output:
                print("%s : %s <- %s" % (str(inputs), str(targets), str(outputs[-1])))
        return outputs


def main():
    # xor で学習テスト
    xor_n = NeuralNetwork(2, 2, 1)
    xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xor_targets = [[0], [1], [1], [0]]
    for _ in [0] * 1000:
        xor_n.train(xor_inputs, xor_targets)
    xor_n.test(xor_inputs, xor_targets, print_output=True)


if __name__ == '__main__':
    main()