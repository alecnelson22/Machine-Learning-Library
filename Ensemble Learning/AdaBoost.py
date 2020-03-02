import numpy as np
from csv import reader
import math
import copy
import abDecisionTree


# The keys provide the map to which weights we need to adjust in data
# incorrectClassifiedKeys: [rootAttr, branchAttr, incorrectLabel]
def adjustSampleWeights(data, incorrectClassifiedKeys, say, features):
    for sample in data:
        # find index of rootAttr from features
        index = list(features.keys()).index(incorrectClassifiedKeys[0])
        if sample[index] == incorrectClassifiedKeys[1] and sample[-2] == incorrectClassifiedKeys[2]:
            sample[-1] = sample[-1] * math.exp(say)
        else:
            sample[-1] = sample[-1] * math.exp(-say)
    return data


def calcError(data, rootIndex, branch, outcome):
    error = 0
    for sample in data:
        if sample[rootIndex] == branch and sample[-2] == outcome:
            error += float(sample[-1])
    return error


# Calculate the amount of say a stump has in the final classification
def amountOfSay(stump):
    return .5*math.log((1-totalError)/totalError)


def normalizeWeights(data):
    totalWeight = 0
    for sample in data:
        totalWeight += float(sample[-1])
    runningWeight = 0
    weightLookup = {}
    for sample in data:
        n = float(sample[-1])/totalWeight
        sample[-1] = n
        runningWeight += n
        weightLookup[runningWeight] = sample
    return data

# # Find if there are more yes's or no's
# def findMajority(classes):
#     maxRes = 0
#     total = 0
#     for key in classes.keys():
#         total += classes[key]
#         if classes[key] > maxRes:
#             maxRes = classes[key]
#     return (total-maxRes)/total


# AdaBoost algorithm


# Make a new dataset based on the weights of the samples from the previous iteration
def getWeightedSample(data, weightLookup):
    rn = np.random.random()
    s = weightLookup.get(rn, weightLookup[min(weightLookup.keys(), key=lambda k: abs(k-rn))])
    sp = s[:]
    # lower = 0
    # upper = 0
    # for sample in data:
    #     upper += float(sample[-1])
    #     if rn > lower and rn < upper:
    #         s = sample[:]
    #         break
    #     else:
    #         lower = upper
    return sp


# Add sample weight value to each data sample
def assignSampleWeights(data):
    num_samples = len(data)
    for sample in data:
        sample.append(1/num_samples)
    return data

#AdaBoost algorithm
def adaBoost(newTrainData, features, originalFeatures):
    stump = abDecisionTree.buildTree(features, originalFeatures, newTrainData, None, None, None, newTrainData, None, depth=0, maxDepth=1)

    # Calculate 'weight' of stump
    root = list(stump.keys())[0]
    rootIndex = list(features.keys()).index(root)
    totalError = 0
    for key in stump[root].keys():
        branch = key
        if stump[root][key] == 'yes':
            error = calcError(newTrainData, rootIndex, branch, 'no')
            totalError += error
        elif stump[root][key] == 'no':
            error = calcError(newTrainData, rootIndex, branch, 'yes')
            totalError += error

    totalError += .0000001
    # This is the 'weight' of our stump
    stumpWeight = .5*math.log((1-totalError-.0000001)/totalError)

    # Adjust the weights in our data
    for key in stump[root].keys():
        branch = key
        if stump[root][key] == 'yes':
            newTrainData = adjustSampleWeights(newTrainData, [root, branch, 'no'], stumpWeight, features)
        elif stump[root][key] == 'no':
            newTrainData = adjustSampleWeights(newTrainData, [root, branch, 'yes'], stumpWeight, features)

    newTrainData = normalizeWeights(newTrainData)
    return newTrainData, stump, stumpWeight
