import os
import numpy as np
from datetime import date
from ML.Model5M_T1 import Model5MT1
from time import clock
import config
import Common.Visualization as v

MODEL_NAME = 'model5m-t1-1.h5'
# 训练代码
STOCK_CODE = 'sh600088'
START_DATE = date(2013, 12, 20)
END_DATE = date(2016, 1, 5)

m5m = Model5MT1(MODEL_NAME)
print(config.PROJECT_ROOT)

start_ts = clock()
X, y = m5m.prepare_data(STOCK_CODE, START_DATE, END_DATE, use_cache=True)
finish_ts = clock()
print("\nExecution time: {:10.6} s".format(finish_ts - start_ts))

print(X[1].shape)
labels = m5m.data_features();
v.animate_data(X, labels, STOCK_CODE)

exit(0)
