import numpy as np
from csv import reader
import math
import copy


# Use entropy as purity function for calculating gain
def entropy(attr, nAttr):
    e = (-attr/nAttr) * math.log(attr/nAttr, 2)
    return e

# Use majority error as purity function for calculating gain
def majorityError(classes):
    maxRes = 0
    total = 0
    for key in classes.keys():
        total += classes[key]
        if classes[key] > maxRes:
            maxRes = classes[key]
    return (total-maxRes)/total

# Use gini index as purity function for calculating gain
def giniIndex(attr, nAttr):
    e = (attr/nAttr)**2
    return e

# Use one of the purity functions above to calculate gain
def calcGain(features, outcomes, purityFn):
    totalEntropy = 0
    totalClasses = 0
    for key in outcomes.keys():
        totalClasses += outcomes[key]
    if purityFn.__name__ == 'majorityError':
        totalEntropy = purityFn(outcomes)
    else:
        for key in outcomes.keys():
            totalEntropy += purityFn(outcomes[key], totalClasses)
        if purityFn.__name__ == 'giniIndex':
            totalEntropy = 1 - totalEntropy

    # Calculate gain for every attribute
    eAttr = {}
    iAttr = {}
    gAttr = {}
    for key in features.keys():
        eAttr[key] = {}
        iAttr[key] = 0
        gAttr[key] = 0
        for key1 in features[key].keys():
            nClasses = 0
            for key2 in features[key][key1]:
                nClasses += features[key][key1][key2]
            if purityFn.__name__ == 'majorityError':
                eAttr[key][key1] = purityFn(features[key][key1])
            else:
                eAttr[key][key1] = 0
                for key2 in features[key][key1]:
                    eAttr[key][key1] += purityFn(features[key][key1][key2], nClasses)
                if purityFn.__name__ == 'giniIndex':
                    eAttr[key][key1] = 1 - eAttr[key][key1]

            # Calculate average information for every attribute
            iAttr[key] += (nClasses/totalClasses) * eAttr[key][key1]

            # Calculate gain for every attribute
            gAttr[key] = totalEntropy - iAttr[key]
    return gAttr

# Probe the data to locate the columns with numerical attributes
def setThreshold(trainData):
    numericalIndices = []
    for i in range(len(trainData[0])):
        if trainData[0][i][-1].isdigit():
            numericalIndices.append(i)
    # Set the thresholds
    numericalMedians = {}
    for line in trainData:
        for i in range(len(line)):
            if i in numericalIndices:
                if i not in numericalMedians.keys():
                    numericalMedians[i] = []
                else:
                    numericalMedians[i].append(int(line[i]))
    for key in numericalMedians.keys():
        numericalMedians[key].sort()
        middle = len(numericalMedians[key])//2
        numericalMedians[key] = numericalMedians[key][middle]
    return numericalMedians

# Convert our numerical attributes to binary ones
def setBinary(data, thresholds):
    newData = copy.deepcopy(data)
    for line in newData:
        for i in range(len(line)):
            if i in thresholds.keys():
                if int(line[i]) < thresholds[i]:
                    line[i] = '0'
                else:
                    line[i] = '1'
    return newData

# Find the most common attr in data
def findMostCommon(data):
    mostCommon = {}
    for i in range(len(data)):
        for j in range(len(data[i])):
            if j not in mostCommon.keys():
                mostCommon[j] = {}
            if data[i][j] == 'unknown':
                continue
            if data[i][j] not in mostCommon[j].keys():
                mostCommon[j][data[i][j]] = 1
            else:
                mostCommon[j][data[i][j]] += 1
    maxAttr = []
    for key in mostCommon.keys():
        maxi = 0
        for key2 in mostCommon[key]:
            if mostCommon[key][key2] > maxi:
                maxi = mostCommon[key][key2]
                most = key2
        maxAttr.append(most)
    return maxAttr

# Change unknown data values to most common attribute
def changeUnknowns(data, maxAttr):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] == 'unknown':
                data[i][j] = maxAttr[j]
    return data

# Load a CSV file
def loadCSV(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    return dataset

# Build decision tree
def buildTree(features, OriginalFeatures, data, parent, clss, originalData, baseFeatures, prevBest, depth=0, maxDepth=16):
    depth += 1
    # Chop the tree if the maxDepth is exceeded
    if depth > maxDepth:
        maxRes = 0
        for key in baseFeatures[prevBest][parent].keys():
            if baseFeatures[prevBest][parent][key] > maxRes:
                maxRes = baseFeatures[prevBest][parent][key]
                maxClass = key
        return maxClass

    features2 = copy.deepcopy(features)
    for key in features2.keys():
        features2[key] = {}

    # exit criteria
    if len(features) == 0:
        return clss

    # exit criteria
    if len(data) == 0:
        countClass = {}
        maxResult = 0
        for key in baseFeatures[prevBest][parent].keys():
            if key not in countClass:
                countClass[key] = baseFeatures[prevBest][parent][key]
            else:
                countClass[key] += baseFeatures[prevBest][parent][key]
            if countClass[key] >= maxResult:
                maxResult = countClass[key]
                returnClass = key
        return returnClass

    outcomes = {}
    for line in data:
        condition = line[-2]
        if condition not in outcomes.keys():
            #outcomes[condition] = 1
            outcomes[condition] = line[-1]
        else:
            #outcomes[condition] += 1
            outcomes[condition] += line[-1]

    # exit criteria
    if len(outcomes.keys()) == 1:
        return list(outcomes.keys())[0]
    if len(outcomes.keys()) == 0:
        return parent

    # Calculate gain
    # In the third argument you may specify which purity function to use
    gAttr = calcGain(features, outcomes, giniIndex)

    # Pick highest gain attribute
    maxGain = 0
    for key in gAttr.keys():
        if gAttr[key] >= maxGain:
            maxGain = gAttr[key]
            bestFeature = key

    tree = {bestFeature:{}}
    bestFeatureIndex = list(features.keys()).index(bestFeature)
    del features2[bestFeature]

    for f in features[bestFeature].keys():
        newFeatures = copy.deepcopy(features2)
        newData = []
        for line in data:
            if line[bestFeatureIndex] == f:
                newData.append(line)
                clss = line[-2]
                for key in newFeatures.keys():
                    # Fetch index of current key from original feature list
                    originalIndex = list(OriginalFeatures.keys()).index(key)
                    attr = line[originalIndex]
                    if attr not in newFeatures[key].keys():
                        newFeatures[key][attr] = {clss: line[-1]}
                    else:
                        if clss not in newFeatures[key][attr].keys():
                            newFeatures[key][attr][clss] = line[-1]
                        else:
                            newFeatures[key][attr][clss] += line[-1]

        # Recursive calls
        subTree = buildTree(newFeatures, OriginalFeatures, newData, f, clss, originalData, features, bestFeature, depth=depth, maxDepth=maxDepth)
        tree[bestFeature][f] = subTree
    return tree

# Visualize decision tree
def printTree(tree, d = 0):
    if (tree == None or len(tree) == 0):
        print("\t" * d, "-")
    else:
        for key, val in tree.items():
            if (isinstance(val, dict)):
                print("\t" * d, key)
                printTree(val, d+1)
            else:
                print("\t" * d, key, str('(') + val + str(')'))

# Test data in decision tree
def test(data, tree, features):
    if type(tree) == str:
        return tree
    else:
        root = list(tree.keys())[0]
    index = list(features.keys()).index(root)
    attr = data[index]
    # Never seen this case before, return most common class
    if attr not in tree[root].keys():
        if data[0][0] == 'vhigh':
            return 'unnac'
        else:
            return 'no'
    subtree = tree[root][attr]
    tree = test(data, subtree, features)
    return tree

