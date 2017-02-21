import os
import numpy as np
from datetime import date
from ML.Model5M_T1 import Model5MT1

MODEL_NAME = 'model5m-t1-1.h5'
# 训练代码

START_DATE = date(2015, 12, 8)
END_DATE = date(2016, 1, 5)

m5m = Model5MT1(MODEL_NAME)

X, y = m5m.prepare_data('sh600108', START_DATE, END_DATE, use_cache=False)

exit(0)