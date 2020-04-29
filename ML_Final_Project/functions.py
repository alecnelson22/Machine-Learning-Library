import numpy as np
import torch
from csv import reader
from csv import writer


# Load a CSV file
def loadCSV(fname):
    file = open(fname, "rt")
    lines = reader(file)
    dataset = list(lines)
    return dataset

# Adds information from csv to dictionary which contains complete
# simulation output for every timestep for entire mesh
def processData(csv_data, target_data):
    # Extract POR and PRESSURE at each timestep
    newTrainData = [[]]
    idxs = []
    for idx, attr in enumerate(csv_data[0]):
        if attr[0:2] == 'PO' or attr == 'PERMI' or attr[0:2] == 'Pr':
            idxs.append(idx)
            newTrainData[0].append(attr)
    for idx, line in enumerate(csv_data):
        if idx == 0:
            continue
        newLine = []
        for i in idxs:
            newLine.append(line[i])
        newTrainData.append(newLine)
    for idx, line in enumerate(newTrainData):
        if idx == 0:
            continue
        for t in range(64):
            target_data[t][idx - 1].append([line[0], line[1], line[t + 2]])
    return target_data

# Write csv
def writeCSV(fname, data):
    print('Writing new csv file...')
    with open(fname, 'w', newline='') as csv_out:
        csv_writer = writer(csv_out)
        for t in range(64):
            csv_writer.writerow(['TIMESTEP_' + str(t)])
            for c in range(13600):
                csv_writer.writerow(data[t][c])