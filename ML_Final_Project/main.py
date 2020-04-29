import numpy as np
import functions
import xgboost as xgb
import csv
from matplotlib import pyplot as plt

n_r1h1 = 39
n_r1h2 = 42
n_r1h3 = 39
num_predictions = 0
num_null_preds = 0
RMSEs = []

# PROCESS TRAIN DATA
# Initialize a dictionary for every simulation timestep for train data
train_timesteps = {}
for t in range(64):
    train_timesteps[t] = {}
    # Create a dictionary for each cell
    for c in range(13600):
        train_timesteps[t][c] = []

# Read in training csv files
print('Reading csv train files to memory...')
for f in range(1,33):
    # Import training data
    fname = 'data/r1h1/complete/r1h1_' + str(f) + '.csv'
    print(fname)
    trainData = functions.loadCSV(fname)
    train_timesteps = functions.processData(trainData, train_timesteps)
print('\n')

# PROCESS TEST DATA
# Create a dictionary for each timestep
test_timesteps = {}
for t in range(64):
    test_timesteps[t] = {}
    # Create a dictionary for each cell
    for c in range(13600):
        test_timesteps[t][c] = []

# Read in test csv files
ignore_idx_arr = []
print('Reading csv test files to memory...')
for f in range(33,40):
    # Import training data
    ignore_idxs = np.load('data/r1h1/complete/ignore_idxs_' + str(f) + '.npy')
    ignore_idx_arr.append(ignore_idxs)
    fname = 'data/r1h1/complete/r1h1_' + str(f) + '.csv'
    print(fname)
    testData = functions.loadCSV(fname)
    test_timesteps = functions.processData(testData, test_timesteps)
print('\n')

# Optionally write to a csv
# functions.writeCSV('data/r1h1/r1h1_total.csv', timesteps)

# Train/test a model for every cell,timestep in the simulation mesh
print('Training/testing XGBoost model for every cell in mesh...')
plot_train = 0
zero_truth = 0
cell_errors = []
avg_errors = []
ii = 0
minPOR = 1
maxPOR = 0
minPERM = 1
maxPERM = 0
# Iterate for every timestep for every cell
for time in range(1,64):
    print('Time = ' + str(time))
    for cell in range(13600):
        zt = False
        # Put train data into DMatrix
        newTrainData = []
        newTrainLabels = []
        for realization in train_timesteps[time][cell]:

            # if float(realization[0]) > maxPOR:
            #     maxPOR = float(realization[0])
            # if float(realization[0]) < minPOR:
            #     minPOR = float(realization[0])
            # if float(realization[1]) > maxPERM:
            #    maxPERM = float(realization[1])
            # if float(realization[1]) < minPERM:
            #     minPERM = float(realization[1])

            newTrainData.append([float(realization[0]), float(realization[1])])
            newTrainLabels.append([float(realization[-1])])
        newTrainData = np.array(newTrainData)
        newTrainLabels = np.array(newTrainLabels)
        train = xgb.DMatrix(newTrainData, newTrainLabels)
        plot_train = train

        # Put test data into DMatrix
        newTestData = []
        newTestLabels = []
        for realization in test_timesteps[time][cell]:
            newTestData.append([float(realization[0]), float(realization[1])])
            newTestLabels.append([float(realization[-1])])
        newTestData = np.array(newTestData)
        newTestLabels = np.array(newTestLabels)
        test = xgb.DMatrix(newTestData, newTestLabels)

        # Train model
        num_round = 40
        max_depth=4
        param = {'bst:max_depth':max_depth, 'bst:eta':1, 'silent':1, 'objective':'reg:squarederror'}
        param['nthread']=1
        plst = param.items()
        bst = xgb.train(plst, train, num_round)

        # Test model, report ccuracy
        # if cell in ignore_idx_arr[0]:
        #     ii += 1
        #     continue

        # Test model, calculate error
        predictions = bst.predict(test)
        residuals = []
        for i in range(len(predictions)):
            num_predictions += 1
            zt = False
            pred = predictions[i]
            # clamp predictions who are less than threshold
            if (pred < 10000):
                pred = 0

            truth = newTestLabels[i][0]
            if (truth == 0):
                if pred > 10000:
                    zt = True
                    zero_truth+=1

            residual = (pred - truth)
            residuals.append(residual)
        avg_cell_residual = np.mean(residuals)
        cell_errors.append(avg_cell_residual)
    # Calculate root mean squared error over entire reservoir
    rmse = np.sqrt(np.mean(np.square(cell_errors)))
    RMSEs.append(rmse)

plt.plot(RMSEs)
plt.title('RMSE over Entire Reservoir')
plt.xlabel('Time')
plt.ylabel('RMSE')
plt.show()

#total_avg_error = sum(avg_errors) / len(avg_errors)
print("Root mean square errors per time t for reservoir: ", RMSEs)
print('Number of predictions: ', num_predictions)
print("Number of non-converged cells: ", zero_truth)

