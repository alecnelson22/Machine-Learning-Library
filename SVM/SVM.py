import numpy as np
import random
from csv import reader
import math
import copy
from scipy.optimize import minimize


# Load a CSV file
def loadCSV(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    return dataset


# Stochastic gradient descent (SGD)
def SGD(X, Y, C, lRate, epochs):
    # initialize weight vector
    w = [0.0 for i in range(len(X[0]))]
    for epoch in range(epochs):
        print('Epoch: ', epoch)
        d = .5
        lRate = lRate / (1 + ((lRate * epoch) / d))
        # lRate = lRate / (1 + epoch)
        # shuffle X, Y based on the same random seed
        s = np.arange(X.shape[0])
        np.random.shuffle(s)
        X = X[s]
        Y = Y[s]
        for i in range(len(X)):
            x = X[i]
            a = hinge(w, x, Y[i], C)
            w = w - (lRate * a)
    print('\n')
    return w


def hinge(w, X, Y, C):
    Y = np.array([Y])
    X = np.array([X])
    loss = 1 - (Y * np.dot(X, w))
    w_new = np.zeros(len(w))
    for i in range(len(loss)):
        if max(0, loss[i]) == 0:
            w_new += w
        else:
            w_new += w - (C * Y[i] * X[i])
    w_new /= len(Y)
    return w_new

def dualSVM(X, Y):
    # Our first order of business is to set up the matrix that is evaluated in the double
    # summation in our minimization problem
    Xlen = len(X)
    XXYY = np.zeros((Xlen, Xlen))
    for i in range(Xlen):
        for j in range(Xlen):
            XXYY[i,j] = np.dot(X[i,:], X[j,:]) * Y[i] * Y[j]
    A = Y[:]
    # These are the bounds on alpha, we set the upper bound to be one of our prescribed C values
    bounds = [(0, 100/873)] * Xlen
    constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(A, alpha), 'jac': lambda alpha: A}
    x0 = np.random.rand(Xlen)
    # Here we use Scipy minimize to minimize our quadratic convex optimization function
    weights = minimize(lambda alpha: .5 * np.dot(alpha.T, np.dot(XXYY, alpha)) - np.sum(alpha), x0,
                    jac=lambda alpha: np.dot(alpha.T, XXYY) - np.ones(alpha.shape[0]),
                    constraints=constraints, method='SLSQP', bounds=bounds)
    return weights

def dualWeights(X, Y, alpha):
    weights = np.array(np.sum(alpha * Y * X.T, 1))
    wTemp = np.zeros(len(weights))
    for i in range(len(weights)):
        wTemp[i] = weights[i]
    bias = np.mean(Y - np.dot(X, wTemp.T))
    coeffs =np.zeros(len(weights) + 1)
    coeffs[:-1] = weights
    coeffs[-1] = bias
    return coeffs

