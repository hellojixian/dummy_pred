import os
import numpy as np
from datetime import date
from ML.Model5M_T1 import Model5MT1
from time import clock
import Common.config as config
import Common.Visualization as v

MODEL_NAME = 'model5m-t1-1.h5'
# 训练代码
training_stock_codes = ['sz000504', 'sh600373', 'sh600088', 'sz002023', 'sz002681', 'sz000596', 'sh600048', 'sz000507',
                        'sh600128', 'sz002166', 'sh600802', 'sz002200', 'sh600847', 'sh600552', 'sh600462']
# ['sh600108', 'sh600246', 'sz002610', 'sh600209', 'sh600131', 'sh600483']
STOCK_CODE = 'sz000839' # sz002352 -均价73  sh600552 - 均价17.3 sh600802 - 均价7   sz002681 - 均价40  # 'sz002166' 这个数据有问题
START_DATE = date(2016, 11, 20)
END_DATE = date(2016, 12, 6)

m5m = Model5MT1(MODEL_NAME)

start_ts = clock()
X, y = m5m.prepare_data(STOCK_CODE, START_DATE, END_DATE, use_cache=True)
finish_ts = clock()
print("\nExecution time: {:10.6} s".format(finish_ts - start_ts))

labels = m5m.data_features();
v.animate_data3d(X, labels, STOCK_CODE)

exit(0)
