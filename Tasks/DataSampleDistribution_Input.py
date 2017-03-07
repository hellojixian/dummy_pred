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
boll_in = input[:, :, [16, 17, 18]]
input = boll_in

y = input.reshape(-1)
y.sort()


v_max = 2
v_min = -2

print("\nraw input range: {} to {}".format(np.min(input), np.max(input)))
print("adjusted range limit: {} to {}".format(v_min, v_max))


# 缩放数据测试
input = y
input = ((input - v_min) / (v_max - v_min)) - 0.5
input = np.tanh(input)
input += 2
# input = input ** 10
y = input



import matplotlib.pyplot as plt
x = range(len(y))
print(len(x), len(y))
fig, ax = plt.subplots(figsize=(10, 8))
plt.title('Sample Distribution BOLL')
ax.grid()
ax.scatter(x=x, y=y, cmap=plt.cm.jet, marker='.')
plt.tight_layout()
plt.show()
