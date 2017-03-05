#!/usr/bin/env python3

'''
NDC - Next day close price change rate
output percentage of changes,
scope: -10% to +10%

TrainingSet is 3D
resultSet is 2D
the first dim is the numerical index
'''

import os, sys, datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

if len(sys.argv) < 2:
    print("{0} start_date end_date".format(sys.argv[0]))
    exit(0)
elif len(sys.argv) == 2:
    start_date = datetime.datetime.strptime(str(sys.argv[1]), "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(str(sys.argv[1]), "%Y-%m-%d").date()
else:
    start_date = datetime.datetime.strptime(str(sys.argv[1]), "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(str(sys.argv[2]), "%Y-%m-%d").date()

import numpy as np
from DataProviders.DailyFullMarket2D import DailyFullMarket2D as Provider
from Models.ModelCNN_NDC import Model_CNN_NDC as Model

low, high, step, samples = -4, 4, 1, 2600
data_segment = 'today_full'
result_cols = ['nextday_close']

provider = Provider(start_date, end_date)
model = Model()

results = provider.fetch_resultset(result_cols)
results = provider.balance_result(result_cols[0], low, high, step, samples)
results = results[result_cols].as_matrix()
results = results[:, 0]
data = provider.fetch_dataset(data_segment)

# results = results * 0.1
# data = data[:10000]
# results = results[:10000]

count = data.shape[0]

[training_data, training_result], \
[validation_data, validation_result], \
[test_data, test_result], \
    = provider.balance_dataset([results, data], low, high, step, 100, 100)

print("Training set size: {}\n"
      "Validation set size: {}\n"
      "Test set size: {}\n".format(training_data.shape[0],
                                   validation_data.shape[0],
                                   test_data.shape[0]))

model.train([training_data, training_result],
            [validation_data, validation_result],
            [test_data, test_result])
