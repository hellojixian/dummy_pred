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

data_segment = 'today_full'
result_cols = ['nextday_close']

provider = Provider(start_date, end_date)

results = provider.fetch_resultset(result_cols)
low, high, step, samples = -7, 7, 1, 2600
# results = provider.balance_result(result_cols[0], low, high, step, samples)
results = results[result_cols].as_matrix()
results = results[:, 0]

dist = np.arange(low, high + step, step)
dist_count = []
results = pd.DataFrame(results)
results.columns = ['value']
report = pd.DataFrame(columns=['range', 'count'])

for i in range(len(dist) - 1):
    low = dist[i]
    high = dist[i + 1]
    test = results.query("value>{} and value<{}".format(low, high))
    count = test.shape[0]
    label = "{}-{}".format(low, high)
    report.loc[i, 'range'] = label
    report.loc[i, 'count'] = count

report = report.set_index("range")
print(report)
print("\nTotal: {} records".format(np.sum(report['count'])))

fig, ax = plt.subplots()
rects1 = plt.bar(np.arange(len(report['count'].index)), tuple(report['count'].values), color='b')
plt.title('Sample Distribution')
plt.xticks(np.arange(len(report['count'].index)), report['count'].index)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=90, fontsize=10)
plt.tight_layout()
plt.show()

