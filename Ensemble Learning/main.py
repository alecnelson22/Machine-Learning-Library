import numpy as np
from csv import reader
import matplotlib.pyplot as plt
import math
import copy
import DecisionTree
import RFDecisionTree
import abDecisionTree
import bonusDecisionTree
import AdaBoost
import Bagging
import RandomForest
import LMS

# Load data
trainData = DecisionTree.loadCSV('data/bank-1/train.csv')
testData = DecisionTree.loadCSV('data/bank-1/test.csv')
features = {'age': {}, 'job': {}, 'marital': {}, 'education': {}, 'default': {}, 'balance': {}, 'housing': {},
            'loan': {}, 'contact': {}, 'day': {}, 'month': {}, 'duration': {}, 'campaign': {}, 'pdays': {},
            'previous': {}, 'poutcome': {}}

# Convert numerical attributes to binary based on median thresholds
numericalMedians = DecisionTree.setThreshold(trainData)
binaryTrainData = DecisionTree.setBinary(trainData, numericalMedians)
testData = DecisionTree.setBinary(testData, numericalMedians)

#============================================
# AdaBoost
#============================================
print('Running AdaBoost for 1 to 10 iterations...')
myAccuracy = []
maxAccuracy = 0
nt = range(1, 10, 1)
for n in nt:
    binaryTrainData1 = copy.deepcopy(binaryTrainData)
    binaryTrainData1 = AdaBoost.assignSampleWeights(binaryTrainData1)

    # Build stump
    stumps = []
    stumpWeights = []
    iterations = n
    newTrainData = binaryTrainData1
    weightLookup = None

    # Run adaBoost algorithm
    for run in range(iterations):
        eFeatures = copy.deepcopy(features)
        # Build dictionary of feature values
        c = 0
        for key in eFeatures.keys():
            for line in newTrainData:
                attr = line[c]
                clss = line[-2]
                if attr not in eFeatures[key].keys():
                    eFeatures[key][attr] = {clss: line[-1]}
                else:
                    if clss not in eFeatures[key][attr].keys():
                        eFeatures[key][attr][clss] = line[-1]
                    else:
                        eFeatures[key][attr][clss] += line[-1]
            c += 1
        newTrainData, stump, stumpWeight = AdaBoost.adaBoost(newTrainData, eFeatures, features)
        stumps.append(stump)
        stumpWeights.append(stumpWeight)

    #Test
    total = 0
    correct = 0
    incorrct = 0
    for sample in testData:
        total += 1
        votes = {'yes': 0, 'no': 0}
        # Run test sample through all trees and tally votes
        for i in range(len(stumps)):
            result = DecisionTree.test(sample, stumps[i], features)
            votes[result] += stumpWeights[i]
        predict = max(votes, key=votes.get)
        actual = sample[-1]
        if predict == actual:
            correct += 1
        else:
            incorrct += 1
    accuracy = correct / total
    if accuracy > maxAccuracy:
        maxAccuracy = accuracy
    myAccuracy.append(accuracy)

fig = plt.figure()
plt.plot(nt, myAccuracy)
plt.title('AdaBoost Test Accuracy')
plt.xlabel('Number of Stumps')
plt.ylabel('Accuracy')
plt.savefig('AdaBoostTest')
plt.show()

#============================================
# Bagging
#============================================
datLen = len(binaryTrainData)
nTrees = 10
nSamples = int(datLen*.6)
oFeatures = copy.deepcopy(features)
trees = []

print('Running Bagged Trees for 1 to 10 iterations...')
myAccuracy = []
nt = range(1, 11, 1)
for n in nt:
    nTrees = n
    trees = []
    for i in range(nTrees):
        trees.append(Bagging.baggedTrees(binaryTrainData, nSamples, oFeatures, datLen))

    #print('Testing Bagged Trees...\n')
    total = 0
    correct = 0
    incorrct = 0
    for line in testData:
        total += 1
        votes = {'yes': 0, 'no': 0}
        # Run test sample through all trees and tally votes
        for tree in trees:
            result = DecisionTree.test(line, tree, features)
            votes[result] += 1
        predict = max(votes, key=votes.get)
        actual = line[-1]
        if predict == actual:
            correct += 1
        else:
            incorrct += 1
    accuracy = correct / total
    myAccuracy.append(accuracy)
    #print('Accuracy: ', accuracy, '\n')

fig = plt.figure()
plt.plot(nt, myAccuracy)
plt.title('Bagged Trees Test Accuracy')
plt.xlabel('Number of trees')
plt.ylabel('Accuracy')
plt.savefig('BaggedTrees')
plt.show()

#============================================
# Random Forest
#============================================
myTrees = {}
myAccuracy = []
nt = range(1, 10, 1)
print('Running Random Forest for 1 to 10 iterations...')
for n in nt:
    nTrees = n
    nf = 2
    rTrees = []
    for i in range(nTrees):
        rTrees.append(RandomForest.randomForest(binaryTrainData, features, nf))
    #print('Random Forest Successfully Generated!\n')

    #print('Testing Random Forest...\n')
    total = 0
    correct = 0
    incorrct = 0
    for line in testData:
        total += 1
        votes = {'yes': 0, 'no': 0}
        # Run test sample through all trees and tally votes
        for tree in rTrees:
            result = RFDecisionTree.test(line, tree, features)
            votes[result] += 1
        predict = max(votes, key=votes.get)
        actual = line[-1]
        if predict == actual:
            correct += 1
        else:
            incorrct += 1
    accuracy = correct / total
    myAccuracy.append(accuracy)
    #print('Accuracy: ', accuracy, '\n')

fig = plt.figure()
plt.plot(nt, myAccuracy)
plt.title('Random Forest Test Accuracy')
plt.xlabel('Number of trees')
plt.ylabel('Accuracy')
plt.savefig('RandomForestT' + str())


#============================================
# Part 2 Problem 2c
#============================================
# predictors = []
# oFeatures = copy.deepcopy(features)
# datLen = len(binaryTrainData)
# nPredictors = 10
# for i in range(nPredictors):
#     trees = []
#     nTrees = 10
#     nSamples = 1000
#     # We will be sampling without replacement here, so we create a copy of our data
#     for i in range(nTrees):
#         dataCopy = binaryTrainData[:]
#         trees.append(Bagging.baggedTrees(dataCopy, nSamples, oFeatures, datLen, replace=False))
#     predictors.append(trees)
#
# myAccuracy = []
# total = 0
# correct = 0
# incorrct = 0
# for line in testData:
#     total += 1
#     votes = {'yes': 0, 'no': 0}
#
#     # Run test sample through FIRST TREE ONLY in all 100 bagged predictors
#     for p in predictors:
#         result = DecisionTree.test(line, p[0], features)
#         votes[result] += 1
#     print(votes)
#     #get ground truth label
#     actual = line[-1]
#
#     if actual == 'yes':
#         predict = votes[actual]/(votes[actual]+votes['no'])
#     elif actual == 'no':
#         predict = votes[actual] / (votes[actual] + votes['yes'])
#
#     bias = (predict-1)**2
#     print(bias)
#
#     #compute mean
#     mean = .5
#     print('mean', mean)
#========
#     if predict == actual:
#         correct += 1
#     else:
#         incorrct += 1
#
# accuracy = correct / total
# myAccuracy.append(accuracy)
# #print('Accuracy: ', accuracy, '\n')
# print(myAccuracy)


#============================================
# Part 3 - Bonus
#============================================
# data = abDecisionTree.loadCSV('data/credit_card.csv')
# features = {'LIMIT_BAL': {}, 'SEX': {}, 'MARRIAGE': {}, 'AGE': {}, 'PAY_0': {}, 'PAY_1': {}, 'PAY_2': {},
#             'PAY_3': {}, 'PAY_4': {}, 'PAY_5': {}, 'PAY_6': {}, 'BILL_AMT1': {}, 'BILL_AMT2': {}, 'BILL_AMT3': {},
#             'BILL_AMT4': {}, 'BILL_AMT5': {}, 'BILL_AMT6': {}, 'PAY_AMT1': {}, 'PAY_AMT2': {}, 'PAY_AMT3': {},
#             'PAY_AMT4': {}, 'PAY_AMT5': {}, 'PAY_AMT6': {}}
# trainData = []
# testData = []
# data.pop(0)
# data.pop(0)
# # Create training and testing sets from full dataset
# for i in range(24000):
#     rn = np.random.randint(0, len(data))
#     data[rn].pop(0)
#     trainData.append(data[rn])
#     data.pop(rn)
# for i in range(6000):
#     rn = np.random.randint(0, len(data))
#     data[rn].pop(0)
#     testData.append(data[rn])
#     data.pop(rn)
#
# # Convert numerical attributes to binary based on median thresholds
# numericalMedians = bonusDecisionTree.setThreshold(trainData)
# binaryTrainData = abDecisionTree.setBinary(trainData, numericalMedians)
# testData =abDecisionTree.setBinary(testData, numericalMedians)
#
# #==========
# # AdaBoost
# #==========
# print('BONUS\n')
# print('Running AdaBoost on credit card dataset for 1 to 10 iterations...')
# myAccuracy = []
# maxAccuracy = 0
# nt = range(1, 10, 1)
# for n in nt:
#     binaryTrainData1 = copy.deepcopy(binaryTrainData)
#     binaryTrainData1 = AdaBoost.assignSampleWeights(binaryTrainData1)
#
#     # Build stump
#     stumps = []
#     stumpWeights = []
#     iterations = n
#     newTrainData = binaryTrainData1
#     weightLookup = None
#
#     # Run adaBoost algorithm
#     for run in range(iterations):
#         eFeatures = copy.deepcopy(features)
#         # Build dictionary of feature values
#         c = 0
#         for key in eFeatures.keys():
#             for line in newTrainData:
#                 attr = line[c]
#                 clss = line[-2]
#                 if attr not in eFeatures[key].keys():
#                     eFeatures[key][attr] = {clss: line[-1]}
#                 else:
#                     if clss not in eFeatures[key][attr].keys():
#                         eFeatures[key][attr][clss] = line[-1]
#                     else:
#                         eFeatures[key][attr][clss] += line[-1]
#             c += 1
#         newTrainData, stump, stumpWeight = AdaBoost.adaBoost(newTrainData, eFeatures, features)
#         stumps.append(stump)
#         stumpWeights.append(stumpWeight)
#
#     #Test
#     total = 0
#     correct = 0
#     incorrct = 0
#     for sample in testData:
#         total += 1
#         votes = {'1': 0, '0': 0}
#         # Run test sample through all trees and tally votes
#         for i in range(len(stumps)):
#             result = bonusDecisionTree.test(sample, stumps[i], features)
#             votes[result] += stumpWeights[i]
#         predict = max(votes, key=votes.get)
#         actual = sample[-1]
#         if predict == actual:
#             correct += 1
#         else:
#             incorrct += 1
#     accuracy = correct / total
#     if accuracy > maxAccuracy:
#         maxAccuracy = accuracy
#     myAccuracy.append(accuracy)
#
# fig = plt.figure()
# plt.plot(nt, myAccuracy)
# plt.title('AdaBoost Test Accuracy')
# plt.xlabel('Number of Stumps')
# plt.ylabel('Accuracy')
# plt.savefig('AdaBoostTest')
# plt.show()

#====================
#====================

# datLen = len(binaryTrainData)
# nTrees = 10
# nSamples = int(datLen*.6)
# oFeatures = copy.deepcopy(features)
# trees = []
#
# myAccuracy = []
# nt = range(1, 11, 1)
# for n in nt:
#     nTrees = n
#     print(str(nTrees) + ' Trees...\n')
#     # print('Generating Random Forest With ' + str(nTrees) + ' Trees...\n')
#     trees = []
#     for i in range(nTrees):
#         trees.append(Bagging.baggedTrees(binaryTrainData, nSamples, oFeatures, datLen))
#     print('Bagged Trees Successfully Generated!\n')
#
#     print('Testing Bagged Trees...\n')
#     total = 0
#     correct = 0
#     incorrct = 0
#     for line in testData:
#         total += 1
#         votes = {'1': 0, '0': 0}
#         # Run test sample through all trees and tally votes
#         for tree in trees:
#             result = bonusDecisionTree.test(line, tree, features)
#             if result == 'no':
#                 print('j')
#             votes[result] += 1
#         predict = max(votes, key=votes.get)
#         actual = line[-1]
#         if predict == actual:
#             correct += 1
#         else:
#             incorrct += 1
#     accuracy = correct / total
#     myAccuracy.append(accuracy)
#     print('Accuracy: ', accuracy, '\n')
# print(myAccuracy)
#
# fig = plt.figure()
# plt.plot(nt, myAccuracy)
# plt.title('Bagged Trees Test Accuracy')
# plt.xlabel('Number of trees')
# plt.ylabel('Accuracy')
# plt.savefig('BaggedTrees')
# plt.show()


#============================================
# Least Mean Squares
#============================================
# Load data
trainData = DecisionTree.loadCSV('data/concrete/train.csv')
testData = DecisionTree.loadCSV('data/concrete/test.csv')

def plotFig(data, title, xlabel, ylabel):
    plt.figure()
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def test(coeff, predictFn):
    testCost = []
    for sample in testData:
        y0 = float(sample[-1])
        y = predictFn(sample, coeff)
        c = ((y - y0) ** 2) / 2
        #c = abs((y-y0)/y0)
        testCost.append(c)
    return testCost

X = []
Y = []
for sample in trainData:
    X.append([float(i) for i in sample[:7]])
    Y.append(float(sample[-1]))
X = np.array(X)
Y = np.array(Y)

B = np.zeros(len(X[1]))
lRate = .01
maxIter = 3000
convThresh = .000001
# Gradient Descent
print('Running Gradient Descent for LMS...\n')
coeff0, pastCost = LMS.BGD(X, Y, lRate, maxIter, convThresh)
print('The learned weight vector is: ', coeff0, '\n')
plotFig(pastCost, 'LMS with Gradient Descent', 'Iterations', 'Cost')
testCost = test(coeff0, LMS.predictG)
plotFig(testCost, 'Predictions with Gradient Descent', 'Sample', 'Cost')

# Stochastic Gradient Descent
print('Running Stochastic Gradient Descent for LMS...\n')
coeff1, pastCost = LMS.SGD(X, lRate, maxIter)
print('The learned weight vector is: ', coeff1, '\n')
plotFig(pastCost, 'LMS with Stochastic Gradient Descent', 'Iterations', 'Cost')
testCost = test(coeff1, LMS.predictG)
plotFig(testCost, 'Predictions with Stochastic Gradient Descent', 'Sample', 'Cost')

# Calculate optimal weight vector analytically
r = np.matmul(X.T, Y)
l = np.linalg.inv(np.matmul(X.T, X))
w = np.matmul(r, l)
print('Optimal weight vector solved analytically: ', w)
testCost = test(w, LMS.predictG)
plotFig(testCost, 'Predictions with Analytical Solution', 'Sample', 'Cost')

