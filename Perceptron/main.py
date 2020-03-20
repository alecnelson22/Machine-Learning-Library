import numpy as np
from csv import reader
import matplotlib.pyplot as plt
import math
import Perceptron


# Load data
trainData = Perceptron.loadCSV('bank-note/train.csv')
testData = Perceptron.loadCSV('bank-note/test.csv')
features = {'variance': {}, 'skewness': {}, 'curtosis': {}, 'entropy': {}}

# Hyperparameters
epochs = 10
lr = .01
nRuns = 10

# Standard Perceptron
avg_acc = 0
print('Running standard perceptron for ' + str(epochs) + ' epochs (' + str(nRuns) + ' times)\n')
for i in range(nRuns):
    weights = Perceptron.standardPerceptron(trainData, lr, epochs)
    acc = Perceptron.testStandard(weights, testData)
    avg_acc += acc
avg_acc /= nRuns
print('Average accuracy: ', avg_acc, '\n')
print('Learned weight vector: ', weights, '\n')

# Voted Perceptron
avg_acc_v = 0
print('Running voted perceptron for ' + str(epochs) + ' epochs (' + str(nRuns) + ' times)\n')
for i in range(nRuns):
    w, c = Perceptron.votedPerceptron(trainData, lr, epochs)
    acc_v = Perceptron.testVoted(w, c, trainData)
    avg_acc_v += acc_v
avg_acc_v /= nRuns
print('Average accuracy (voted): ', avg_acc_v, '\n')
print('Number of weight vectors: ', len(w), '\n')
print('Weights: ', w, '\n')
print('Counts: ', c, '\n')

# Average Perceptron
avg_acc_a = 0
print('Running averaged perceptron for ' + str(epochs) + ' epochs (' + str(nRuns) + ' times)\n')
for i in range(nRuns):
    w, c = Perceptron.votedPerceptron(trainData, lr, epochs)
    acc_a = Perceptron.testAveraged(w, c, trainData)
    avg_acc_a += acc_a
avg_acc_a /= nRuns
print('Average accuracy (averaged): ', avg_acc_a, '\n')
print('Number of weight vectors: ', len(w), '\n')
print('Weights: ', w, '\n')
print('Counts: ', c, '\n')



