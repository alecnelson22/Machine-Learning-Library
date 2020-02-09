import numpy as np
from csv import reader
import math
import copy
import ID3

# Load data
trainData = ID3.loadCSV('data/car/train.csv')
testData = ID3.loadCSV('data/car/test.csv')
features = {'buying': {}, 'maint': {}, 'doors': {}, 'persons': {}, 'lug': {}, 'safety': {}}

# trainData = ID3.loadCSV('data/bank/train.csv')
# testData = ID3.loadCSV('data/bank/test.csv')
# features = {'age': {}, 'job': {}, 'marital': {}, 'education': {}, 'default': {}, 'balance': {}, 'housing': {},
#             'loan': {}, 'contact': {}, 'day': {}, 'month': {}, 'duration': {}, 'campaign': {}, 'pdays': {},
#             'previous': {}, 'poutcome': {}}
#
# # Locate the medians of our numerical attributes
# # We will use this as a binary threshold for our tree
# numericalMedians = ID3.setThreshold(trainData)
# trainData = ID3.setBinary(trainData, numericalMedians)
# testData = ID3.setBinary(testData, numericalMedians)
#
# # Change unknown values in our data to the most common attribute in that column
# maxAttr = ID3.findMostCommon(trainData)
# trainData = ID3.changeUnknowns(trainData, maxAttr)
# testData = ID3.changeUnknowns(testData, maxAttr)

# Build dictionary of feature values
c = 0
for key in features.keys():
    for line in trainData:
        attr = line[c]
        clss = line[-1]
        if attr not in features[key].keys():
            features[key][attr] = {clss: 1}
        else:
            if clss not in features[key][attr].keys():
                features[key][attr][clss] = 1
            else:
                features[key][attr][clss] += 1
    c += 1

# # Build trees of increasing depths and put them in array
# Trees = []
# for i in range(1, 17):
#     myTree = ID3.buildTree(features, features, trainData, None, None, None, trainData, None, depth=0, maxDepth=i)
#     Trees.append(myTree)

myTree = ID3.buildTree(features, features, trainData, None, None, None, trainData, None, depth=0, maxDepth=6)
ID3.printTree(myTree)

# # Test data in decision trees of varying depth
# results = []
# for i in range(1, 17):
#     total = 0
#     correct = 0
#     incorrct = 0
#     for line in trainData:
#         total += 1
#         result = ID3.test(line, Trees[i - 1], features)
#         actual = line[-1]
#         if result == actual:
#             correct += 1
#         else:
#             incorrct += 1
#     accuracy = correct / total
#     results.append(accuracy)
# print('Accuracy: ', results)

# Test data for single tree
total = 0
correct = 0
incorrct = 0
for line in testData:
    total += 1
    result = ID3.test(line, myTree, features)
    actual = line[-1]
    if result == actual:
        correct +=1
    else:
        incorrct += 1
accuracy = correct/total
print('Accuracy: ', accuracy)