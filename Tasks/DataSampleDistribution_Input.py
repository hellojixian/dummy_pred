#!/usr/bin/env python3

import os, sys, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

from DataProviders.DailyFullMarket2D import DailyFullMarket2D as Provider

low, high, step, samples = -9.5, 9.5, 1, 3500
data_segment = 'today_full'
result_cols = ['nextday_close']

provider = Provider(start_date, end_date, [])

cond = " `{0}` > {1} AND `{0}` < {2} ".format(result_cols[0], low, high)

results = provider.fetch_resultset(result_cols, cond)
results = provider.balance_result(result_cols[0], low, high, step, samples)
results = results[result_cols].as_matrix()
results = results[:, 0]
input = provider.fetch_dataset(data_segment)

# 先不缩放数据 只是观察
# psy_in = input[:, :, [68, 69]]
# amp_in = input[:, :,  [20]] #vr 22 vol 21
mdi_in = input[:, :,  [57, 58, 59, 60]]
input = mdi_in
input = np.nan_to_num(input)

y = input.reshape(-1)
y.sort()

# # vol 21
# v_max = -1
# v_min = -4.5

# vol 21
v_max = 5
v_min = 0


print("\nraw input range: {} to {}".format(np.min(input), np.max(input)))
print("adjusted range limit: {} to {}".format(v_min, v_max))

# 缩放数据测试
c = y
input = y
input = ((input - v_min) / (v_max - v_min)) - 0.5
input = np.tanh(input)
input += 2
y = input

#
# c = ((c - -5) / (5 - -5)) - 0.5
# c = np.tanh(c)
# c += 2

print("\nscaled input range: {} to {}".format(np.min(y), np.max(y)))
import matplotlib.pyplot as plt

x = range(len(y))
print(len(x), len(y))
fig, ax = plt.subplots(figsize=(10, 8))
plt.title('Sample Distribution CCI')
ax.grid()
ax.scatter(x=x, y=y, c='b', marker='.')
# ax.scatter(x=x, y=c, c='r', marker='.')
plt.tight_layout()
plt.show()
