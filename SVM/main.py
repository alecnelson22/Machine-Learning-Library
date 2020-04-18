import numpy as np
from csv import reader
import matplotlib.pyplot as plt
import math
import SVM
import scipy

# Load data
trainData = SVM.loadCSV('data/bank-note/train.csv')
testData = SVM.loadCSV('data/bank-note/test.csv')
features = {'variance': {}, 'skewness': {}, 'curtosis': {}, 'entropy': {}}

# Convert class values to be in range [-1, 1]
for sample in trainData:
    if sample[-1] == '0':
        sample[-1] = '-1'
for sample in testData:
    if sample[-1] == '0':
        sample[-1] = '-1'

# Convert training/test data into usable form for SGD
X_train = []
Y_train = []
X_test = []
Y_test = []
for sample in trainData:
    X_train.append([float(i) for i in sample[:4]])
    Y_train.append(float(sample[-1]))
for sample in testData:
    X_test.append([float(i) for i in sample[:4]])
    Y_test.append(float(sample[-1]))
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# Hyperparameters
epochs = 0
lRate = .01
C = ['100/873', '500/873', '700/873']
epochs = 100

print('Running Stochastic Gradient Descent for SVM...\n')
#while(epochs < maxIter):

lRate = .01
for c in C:
    w = SVM.SGD(X_train, Y_train, eval(c), lRate, epochs)
    # Get predictions from model
    Y_predict = []
    correct = 0
    n = len(X_train)
    for i in range(n):
        yp = np.sign(np.dot(w, X_train[i]))
        Y_predict.append(yp)
    # Calculate accuracy
    for i in range(len(Y_train)):
        if Y_train[i] == Y_predict[i]:
            correct += 1
    accuracy = correct / n
    print('C = ', c)
    print('Learned weight vector: ', w)
    print('Accuracy: ', accuracy, '\n')

# Dual SVM
# Note that for each C value, scipy.minimize could take ~1 minute to run
print('Running Dual SVM...')
print('(Be patient, let scipy.minimize work its magic!)\n')
for c in C:
    # Train
    sln = SVM.dualSVM(X_train, Y_train)
    support_vectors = sln.x > 0
    weights = SVM.dualWeights(X_train[support_vectors,:], Y_train[support_vectors], sln.x[support_vectors])
    Y_predict = []
    correct = 0
    n = len(X_test)
    # Test
    for i in range(n):
        yp = np.sign(np.dot(weights[:-1], X_test[i]) + weights[-1])
        Y_predict.append(yp)
    # Calculate accuracy
    for i in range(len(Y_test)):
        if Y_test[i] == Y_predict[i]:
            correct += 1
    accuracy = correct / n
    print('C = ', c)
    print('Learned weight vector: ', weights)
    print('Accuracy: ', accuracy, '\n')
