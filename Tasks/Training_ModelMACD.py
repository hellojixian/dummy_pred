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

import Common.config as config
import numpy as np
import tushare as ts
import pandas as pd

cache_file = os.path.join(config.CACHE_DIR, 'stock_list_sz50.csv')
if os.path.isfile(cache_file):
    sample_stock_list = pd.read_csv(cache_file)
else:
    sample_stock_list = ts.get_sz50s()
    sample_stock_list.to_csv(cache_file, index=False)

raw_stock_list = sample_stock_list['code'].tolist()
stock_list = []
stocks_str = []
for code in raw_stock_list:
    code = str(code)
    if len(code) == 6 \
            and (code[:2] == '60' \
                         or code[:2] == '00' \
                         or code[:2] == '30'):
        if code[:2] == '60':
            stock_list.append("sh" + code)
        else:
            stock_list.append("sz" + code)

from DataProviders.DailyFullMarket2D import DailyFullMarket2D as Provider
from Models.ModelMACD_AE import ModelMACD as Model

low, high, step, samples = -9.5, 9.5, 1, 3500
data_segment = 'today_full'
result_cols = ['nextday_close']

provider = Provider(start_date, end_date,[])
model = Model()

cond = " `{0}` > {1} AND `{0}` < {2} ".format(result_cols[0], low, high)

results = provider.fetch_resultset(result_cols, cond)
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
    = provider.balance_dataset([results, data], low, high, step,
                               validation_samples_ratio=0.05,
                               test_samples_ratio=0.05)

print("Total data set size: {}\n"
      "Training set size: {}\n"
      "Validation set size: {}\n"
      "Test set size: {}\n".format(count, training_data.shape[0],
                                   validation_data.shape[0],
                                   test_data.shape[0]))

print("training result max:{} min:{}".format(round(np.max(training_result),2), round(np.min(training_result)),2))

model.train([training_data, training_result],
            [validation_data, validation_result],
            [test_data, test_result])
