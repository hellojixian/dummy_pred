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

if len(sys.argv) != 3:
    print("{0} start_date end_date".format(sys.argv[0]))
    exit(0)

import numpy as np
from DataProviders.DailyFullMarket2D import DailyFullMarket2D as Provider
from Models.ModelCNN_NDC import Model_CNN_NDC as Model

data_ratio = [0.85, 0.1, 0.15]
data_segment = 'today_full'
result_cols = ['nextday_close']

start_date = datetime.datetime.strptime(str(sys.argv[1]), "%Y-%m-%d").date()
end_date = datetime.datetime.strptime(str(sys.argv[2]), "%Y-%m-%d").date()

provider = Provider(start_date, end_date)
model = Model()

data = provider.fetch_dataset(data_segment)
results = provider.fetch_resultset(result_cols)
results = results[result_cols].as_matrix()
results = results[:, 0]


# data = data[:1000]
# results = results[:1000]

count = data.shape[0]

splitter = round(count * data_ratio[0])
training_set = data[:splitter]
training_result = results[:splitter]

splitter = round(count * data_ratio[1])
validation_set = data[:splitter]
validation_result = results[:splitter]

splitter = round(count * data_ratio[2])
test_set = data[:splitter]
test_result = results[:splitter]

model.train([training_set, training_result],
            [validation_set, validation_result],
            [test_set, test_result])
