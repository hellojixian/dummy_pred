#!/usr/bin/env python3

import os, sys, datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

import Common.config as config
import pandas as pd
from sqlalchemy.orm import sessionmaker
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange

# 参数设置
start_date = datetime.date(2016, 1, 1)
end_date = datetime.date(2016, 12, 30)
code = 'sz002166'
col = 'close'
data_splitter = 0.715

features = ['close', 'vol', 'count']
timesteps = 10
prediction_step = 4

ds_cache_file = os.path.join(config.CACHE_DIR,'LSTMResearchDataset.csv')
rs_cache_file = os.path.join(config.CACHE_DIR,'LSTMResearchResult.csv')

if os.path.isfile(ds_cache_file):
    ds_df = pd.read_csv(ds_cache_file)
    dataset = ds_df.values
    dataset = dataset.reshape(dataset.shape[0], timesteps, len(features))

    rs_df = pd.read_csv(rs_cache_file)
    result = rs_df
else:
    # 获取数据
    session = sessionmaker()
    session.configure(bind=config.DB_CONN)
    db = session()
    sql = """
    SELECT
        `time`,
        `open`,
        `high`,
        `low`,
        `close`,
        `vol`,
        `count`
    FROM
        feature_extracted_stock_trading_5min
    WHERE
        `code` = '{0}'
            AND `time` > '{1}'
            AND `time` < '{2}' ;
    """.format(code, start_date, end_date)
    rs = db.execute(sql)
    df = pd.DataFrame(rs.fetchall())
    df.columns = ['time', 'open', 'high', 'low', 'close', 'vol', 'count']
    db.close()

    # 准备训练数据

    dataset = np.zeros((df.shape[0], timesteps, len(features)))
    result = pd.DataFrame(np.zeros([df.shape[0], 2]))
    i = 0
    for time, row in df.iterrows():
        if i >= timesteps:
            for t in range(timesteps):
                step = timesteps - t - 1
                rec = df.iloc[i - step, :]
                dataset[i, t] = rec[features].values

        if i < df.shape[0] - prediction_step:
            next_rec = df.iloc[i + prediction_step, :]
            result.iloc[i, 0] = next_rec[col]
            result.iloc[i, 1] = next_rec['time']
        i += 1
        pass

    result.columns = [col, 'time']
    result.index = result['time']
    result = result.drop('time', axis=1)

    # 掐头
    # df = df[timesteps:]
    dataset = dataset[timesteps:]
    result = result[timesteps:]

    # 去尾
    shift_size = dataset.shape[0] - prediction_step
    # df = df[:shift_size]
    dataset = dataset[:shift_size]
    result = result[:shift_size]

    # 缓存
    ds_df = pd.DataFrame(dataset.reshape(dataset.shape[0], -1))
    ds_df.to_csv(path_or_buf=ds_cache_file, index=False)

    rs_df = result
    rs_df.to_csv(path_or_buf=rs_cache_file, index=False)

sep_pt = round(dataset.shape[0] * data_splitter)

print("dataset: {}\t result:{}".format(dataset.shape[0], result.shape[0]))
print("sep_pt: {}".format(sep_pt))


# 整理训练数据集
training_X = dataset[:sep_pt]
test_X = dataset[sep_pt:]

training_y = result[:sep_pt]
test_y = result[sep_pt:]

# 图形输出
fig, ax = plt.subplots(figsize=(16, 8))
plt.title("5 mins K-Chart for {0} from {1} to {2}".format(code, start_date, end_date))

# 分割线
sep_max, sep_min = np.max(result) - (np.max(result) - np.min(result)), np.max(result)
sep_line = plt.plot(np.repeat(sep_pt - 1, 2), (sep_max, sep_min), color='black')

# 原始数据
training_display = training_y.values.reshape(-1)

prediction_display = test_y.values.reshape(-1)
# 用真实数据的最后一个来补位
# prediction_prepend = training_y.reshape(-1)
# prediction_display = np.hstack((prediction_prepend[prediction_prepend.shape[0]-prediction_step:], prediction_display))


raw_line = plt.plot(range(result.shape[0]), result.values, color='b')

training_line = plt.plot(np.arange(sep_pt),
                         training_display,
                         color='lime', marker="o")
test_line = plt.plot(np.arange(sep_pt, sep_pt + prediction_display.shape[0]),
                     prediction_display, color='r', marker="o")

plt.xticks(np.arange(0,result.shape[0],20), rotation=45, fontsize=10)

plt.tight_layout()
plt.show()
