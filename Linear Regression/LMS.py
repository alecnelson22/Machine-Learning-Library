import numpy as np
from csv import reader
import math
import copy


# Make a prediction for SGD
def predictS(sample, coeff):
    y = coeff[0]
    for i in range(len(sample)-1):
        y += coeff[i+1] * float(sample[i])
    return y

# Make a prediction for GD
def predictG(sample, coeff):
    y = 0
    for i in range(len(sample)-1):
        y += coeff[i] * float(sample[i])
    return y

# Batch gradient descent
def BGD(X, Y, lRate, maxIter, convThresh):
    pastCost = []
    coeff = [0.0 for i in range(len(X[0]))]
    for i in range(maxIter):
        loss = X.dot(coeff) - Y
        grad = X.T.dot(loss) / len(Y)
        coeff = coeff - lRate * grad
        cost = np.sum((X.dot(coeff) - Y) ** 2)/(2 * len(Y))
        # Check if convergence criteria is satisfied
        if i != 0:
            if abs(cost - pastCost[-1]) < convThresh:
                return coeff, pastCost
        pastCost.append(cost)
    return coeff, pastCost


# Stochastic gradient descent
def SGD(X, lRate, maxIter):
    coeff = [0.0 for i in range(len(X[0]))]
    pastCost = []
    for i in range(maxIter):
        cost = 0
        rn = np.random.randint(0, len(X))
        sample = X[rn]
        y = predictS(sample, coeff)
        cost += ((y - sample[-1])**2)/2
        coeff[0] = coeff[0] - lRate * (y - sample[-1])
        for j in range(len(sample)-1):
            coeff[j + 1] = coeff[j + 1] - lRate * (y - sample[-1]) * sample[j]
        pastCost.append(cost)
    return coeff, pastCost