# import DataLoader.DailyDataLoader as DataLoader
# DataLoader.load_data_to_db()

# import DataLoader.MinutesDataLoader as DataLoader
# DataLoader.load_data_to_db()

import os
import numpy as np
from datetime import date
from ML.Model5M_T1 import Model5MT1

MODEL_NAME = 'model5m-t1.h5'
# 训练代码
training_stock_codes = ['sz002023', 'sz002681', 'sz000596', 'sh600048', 'sz000507', 'sh600128',
                        'sz002166', 'sh600802', 'sz002200', 'sh600847', 'sh600552', 'sh600462']
validation_stock_codes = ['sh601908', 'sh600857', 'sz002143', 'sz002595', 'sz300176', 'sz002610', 'sh600452']
test_stock_codes = ['sh600108', 'sh600246', 'sz002610', 'sh600209', 'sh600131', 'sh600483']
START_DATE = date(2013, 1, 8)
END_DATE = date(2016, 1, 5)
TESTING_START_DATE = date(2015, 10, 12)
TEST_SAMPLES = 100

m5m = Model5MT1(MODEL_NAME)

# 训练模型 准备数据
X_list, y_list = [], []
for code in training_stock_codes:
    X, y = m5m.prepare_data(code, START_DATE, END_DATE, use_cache=True)
    X_list.append(X)
    y_list.append(y)
    print("\nStock code:", code)
    print("{0} Samples distributions: ".format(X.shape[0]))
    print(np.sum(y, axis=0))
training_set = (np.vstack(X_list), np.vstack(y_list))

X_list, y_list = [], []
for code in validation_stock_codes:
    X, y = m5m.prepare_data(code, TESTING_START_DATE, END_DATE, use_cache=True)
    X_list.append(X)
    y_list.append(y)
    print("\nStock code:", code)
    print("{0} Samples distributions: ".format(X.shape[0]))
    print(np.sum(y, axis=0))
validation_set = (np.vstack(X_list), np.vstack(y_list))

X_list, y_list = [], []
for code in test_stock_codes:
    X, y = m5m.prepare_data(code, TESTING_START_DATE, END_DATE, use_cache=True)
    X_list.append(X)
    y_list.append(y)
    print("\nStock code:", code)
    print("{0} Samples distributions: ".format(X.shape[0]))
    print(np.sum(y, axis=0))
test_set = (np.vstack(X_list), np.vstack(y_list))

print("\nTraining set: \n"
      "Overall {0} Samples distributions: {1}".format(training_set[0].shape[0], np.sum(training_set[1], axis=0)))
print("\nValidation set: \n"
      "Overall {0} Samples distributions: {1}".format(validation_set[0].shape[0], np.sum(validation_set[1], axis=0)))
print("\nTesting set: \n"
      "Overall {0} Samples distributions: {1}".format(test_set[0].shape[0], np.sum(test_set[1], axis=0)))
print('\n\t')

loss, accuracy = m5m.train(training_set, validation_set, test_set)
print('\ntest loss: ', loss)
print('test accuracy: ', accuracy)
exit(0)
#
# # 使用模型
# stock_codes = ['sh600552',  'sh600313', 'sh600857',  'sh600128']
# START_DATE = date(2015, 10, 14)
# END_DATE = date(2016, 1, 5)
stock_codes = ['sz000797']
START_DATE = date(2015, 5, 5)
END_DATE = date(2016, 1, 25)
X_list, y_list = [], []
for code in stock_codes:
    X, y = m5m.prepare_data(code, START_DATE, END_DATE, use_cache=False)
    X_list.append(X)
    y_list.append(y)
    print("\nStock code:", code)
    print("{0} Samples distributions: ".format(X.shape[0]))
    print(np.sum(y, axis=0))
X = np.vstack(X_list)
y = np.vstack(y_list)

y = np.argmax(y, axis=1)
i = 0
err = 0
for x in X:
    r = m5m.predict(x)
    real_r = y[i]
    print(r[0] == real_r, real_r, r[0], r[1])
    if r[0] != real_r:
        err += 1
    i += 1

print("Err count: {0}  rate:{1}".format(err, np.round(err / X.shape[0] * 100)))
