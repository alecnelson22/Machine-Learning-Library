import numpy as np
from csv import reader
import math
import copy
import RFDecisionTree


# Returns a bootstrapped dataset
def bootstrapData(data):
    bsData = []
    for i in range(len(data)):
        rn = np.random.randint(0, len(data))
        s = data[rn]
        bsData.append(s[:])
    return bsData


# Fetch a random root (as a string) from a feature dictionary
# nf denotes the number of feature values we consider at each step in building our tree
def getRandomFeatures(features, nf):
    newFeatures = {}
    featureList = list(features.keys())
    for i in range(nf):
        rn = np.random.randint(0, len(featureList))
        f = featureList[rn]
        newFeatures[f] = {}
        featureList.pop(rn)
    return newFeatures


# Random forest algorithm
def randomForest(data, features, nf):
    # Randomly select two features on which to build a tree
    # These will both be candidates for the root node
    newFeatures = getRandomFeatures(features, nf)

    # Generate bootstrapped dataset from original data
    bsData = bootstrapData(data)

    # Build dictionary of feature values
    for key in newFeatures.keys():
        attrIndex = list(features.keys()).index(key)
        for line in bsData:
            attr = line[attrIndex]
            clss = line[-1]
            if attr not in newFeatures[key].keys():
                newFeatures[key][attr] = {clss: 1}
            else:
                if clss not in newFeatures[key][attr].keys():
                    newFeatures[key][attr][clss] = 1
                else:
                    newFeatures[key][attr][clss] += 1

    # Determine the best attribute from the previous two features
    OF = copy.deepcopy(features)
    tree = RFDecisionTree.buildTree(newFeatures, features, bsData, None, None, bsData, OF, None, depth=0, maxDepth=20, nf=nf)
    return tree





