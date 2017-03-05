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
import pandas as pd

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
import tushare as ts


from DataProviders.DailyFullMarket2D import DailyFullMarket2D as Provider
from Models.ModelCNN_NDC_relu import Model_CNN_NDC_relu as Model

data_segment = 'today_full'
result_cols = ['nextday_close']

provider = Provider(start_date, end_date)
model = Model()

results = provider.fetch_resultset(result_cols)
data = provider.fetch_dataset(data_segment)
real_results = results[result_cols]
real_results = real_results.as_matrix()[:, 0]

# data = data[:1000]
# real_results = results[:1000]

print("Evaluating {} samples".format(data.shape[0]))
pred_results = model.predict(data)
pred_results = pred_results.reshape(1, -1)

results = pd.DataFrame(results)
results['prediction'] = pd.Series(pred_results[0].tolist())
# results['prediction'] *= 10
results['diff'] = results[result_cols[0]] - results['prediction']

pd.set_option('display.max_rows', results.shape[0])
print(results)

# print("Real:")
# print("mean absolute error: {}".format(np.mean(np.abs(results['diff']))))
# print("max absolute error: {}".format(np.max(results['diff'])))
#

print("Prediction:")
print("mean absolute error: {}".format(np.mean(np.abs(results['diff']))))
print("max absolute error: {}".format(np.max(results['diff'])))
