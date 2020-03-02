import numpy as np
from csv import reader
import math
import copy
import DecisionTree


def baggedTrees(bTrainData, nSamples, oFeatures, datLen, replace=True):
    baggedData = []
    for i in range(nSamples):
        rn = np.random.randint(0, datLen)
        s = bTrainData[rn]
        baggedData.append(s[:])
        # Sample without replacement
        if replace == False:
            datLen -= 1
            bTrainData.pop(rn)


    # Build dictionary of feature values
    features = copy.deepcopy(oFeatures)
    c = 0
    for key in features.keys():
        for line in baggedData:
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

    bTree = DecisionTree.buildTree(features, features, baggedData, None, None, None, baggedData, None, depth=0, maxDepth=16)
    return bTree
