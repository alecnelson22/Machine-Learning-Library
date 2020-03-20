import numpy as np
import random
from csv import reader
import math
import copy


# Load a CSV file
def loadCSV(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    return dataset


# Use weights to make a prediction
def predict(weights, x_i):
    p = weights[0]
    for i in range(len(x_i) - 1):
        p += weights[i + 1] * float(x_i[i])
    if p > 0:
        return 1
    else:
        return 0


def updateWeights(weights, x_i, lr, sign):
    weights[0] = weights[0] + lr * sign
    for i in range(len(x_i)-1):
        weights[i+1] = weights[i+1] + lr * sign * float(x_i[i])

def standardPerceptron(data, lr, epochs):
    weights = [0 for i in range(len(data[0]))]
    for e in range(epochs):
        random.shuffle(data)
        for sample in data:
            p = predict(weights, sample)
            if p == 0 and float(sample[-1]) > 0:
                updateWeights(weights, sample, lr, 1)
            elif p == 1 and float(sample[-1]) == 0:
                updateWeights(weights, sample, lr, -1)
    return weights


def votedPerceptron(data, lr, epochs):
    weights = [0 for i in range(len(data[0]))]
    w = []
    c = []
    c_i = 1
    for e in range(epochs):
        random.shuffle(data)
        for sample in data:
            p = predict(weights, sample)
            if p == 0 and float(sample[-1]) > 0:
                w.append(weights)
                updateWeights(weights, sample, lr, 1)
                c.append(c_i)
                c_i = 1
            elif p == 1 and float(sample[-1]) == 0:
                w.append(weights)
                updateWeights(weights, sample, lr, -1)
                c.append(c_i)
                c_i = 1
            else:
                c_i += 1
    return w, c

def averagePerceptron():
    pass


def testStandard(weights, testData):
    correct = 0
    incorrect = 0
    count = 0
    for sample in testData:
        count += 1
        p = weights[0]
        for i in range(len(sample) - 1):
            p += weights[i + 1] * float(sample[i])
        if p <= 0 and float(sample[-1]) == 0:
            correct += 1
        elif p > 0 and float(sample[-1]) > 0:
            correct += 1
        else:
            incorrect += 1
    accuracy = correct / count
    return accuracy


def testVoted(w, c, testData):
    correct = 0
    incorrect = 0
    count = 0
    for sample in testData:
        count += 1
        pSum = 0
        for i in range(len(w)):
            weights = w[i]
            p = weights[0]
            for j in range(len(sample) - 1):
                p += weights[j + 1] * float(sample[j])
            if p <= 0:
                p = -1
            else:
                p = 1
            p *= c[i]
            pSum += p
        if pSum <= 0 and float(sample[-1]) == 0:
            correct += 1
        elif pSum > 0 and float(sample[-1]) > 0:
            correct += 1
        else:
            incorrect += 1
    accuracy = correct / count
    return accuracy


def testAveraged(w, c, testData):
    correct = 0
    incorrect = 0
    count = 0
    for sample in testData:
        count += 1
        pSum = 0
        for i in range(len(w)):
            weights = w[i]
            p = weights[0]
            for j in range(len(sample) - 1):
                p += weights[j + 1] * float(sample[j])
            p *= c[i]
            pSum += p
        if pSum <= 0 and float(sample[-1]) == 0:
            correct += 1
        elif pSum > 0 and float(sample[-1]) > 0:
            correct += 1
        else:
            incorrect += 1
    accuracy = correct / count
    return accuracy