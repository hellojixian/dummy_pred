#!/usr/bin/env python3

import os, sys, datetime, keras

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

import Common.config as config
import pandas as pd
from sqlalchemy.orm import sessionmaker
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Convolution2D, BatchNormalization, Merge, LSTM, GRU
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
from Common.KerasMetrics import root_mean_squared_error as rmse

# 参数设置
start_date = datetime.date(2017, 1, 1)
end_date = datetime.date(2017, 1, 30)
code = 'sz002166'
col = 'close'
data_splitter = 0.8
test_splitter = 0.5

features = ['open', 'low', 'high', 'close', 'ma5', 'ma15', 'ma25', 'ma40',
            'ema_5', 'ema_15', 'ema_25', 'ema_40']
timesteps = 10  # 过去两个小时的走势
prediction_step = 8  #
batch_size = 48  # 一天有48个五分钟

model_weight_file = os.path.join(config.MODEL_DIR, 'LSTMResearchNextCloseS8_weight.h5')

np.random.seed(7)

# 获取数据
session = sessionmaker()
session.configure(bind=config.DB_CONN)
db = session()

columns_list = ['time'] + features
columns_str = '`' + "`,`".join(columns_list) + '`'
sql = """
SELECT {3}
FROM
    feature_extracted_stock_trading_5min
WHERE
    `code` = '{0}'
        AND `time` > '{1}'
        AND `time` < '{2}' ;
""".format(code, start_date, end_date, columns_str)
rs = db.execute(sql)
df = pd.DataFrame(rs.fetchall())
df.columns = columns_list
db.close()

# 图形输出
fig, ax = plt.subplots(figsize=(16, 8))
plt.title("5 mins K-Chart for {0} from {1} to {2}".format(code, start_date, end_date))

# 分割线
sep_min, sep_max = np.max(df[col].values) - (np.max(df[col].values) - np.min(df[col].values)), np.max(df[col].values)
sep_line = plt.plot(np.repeat(0, 2), (sep_max, sep_min), color='black')
sep_text = plt.text(0, int((sep_max + sep_min) / 2), "sep")
major_ticks = np.arange(np.round(sep_min), np.round(sep_max), 0.5)
minor_ticks = np.arange(np.round(sep_min), np.round(sep_max), 0.1)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.set_ylim(sep_min, sep_max)
ax.set_yticklabels(minor_ticks, minor=True)

# result


raw_line = plt.plot(range(df.shape[0]), df[col].values, color='b', alpha=0.5)
past_line = plt.plot([0], [df[col].values[0]], color='b', alpha=1, marker='.')

pred_line = plt.plot([0], [df[col].values[0]], color='r', alpha=1, marker='.')
past_pred_line = plt.plot([0], [df[col].values[0]], color='r', alpha=0.5, marker='.')

plt.xticks(np.arange(0, df.shape[0], 100), rotation=45, fontsize=10)
plt.ion()
plt.tight_layout()
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
# plt.ioff()
# plt.show()
# exit()
plt.pause(0.5)

# 准备预测模型
model = Sequential([
    LSTM(100,
         batch_input_shape=(batch_size, timesteps, len(features)),
         return_sequences=True,
         stateful=False,
         init='glorot_uniform',
         inner_init='orthogonal',
         forget_bias_init='one',
         activation='tanh',
         inner_activation='hard_sigmoid',
         W_regularizer=None,
         U_regularizer=None,
         b_regularizer=None,
         dropout_W=0.0,
         dropout_U=0.0,
         name="lstm_1"),
    LSTM(50,
         batch_input_shape=(batch_size, timesteps, len(features)),
         return_sequences=True,
         stateful=False,
         init='glorot_uniform',
         inner_init='orthogonal',
         forget_bias_init='one',
         activation='tanh',
         inner_activation='hard_sigmoid',
         W_regularizer=None,
         U_regularizer=None,
         b_regularizer=None,
         dropout_W=0.0,
         dropout_U=0.0,
         name="lstm_2"),
    LSTM(30,
         batch_input_shape=(batch_size, timesteps, len(features)),
         return_sequences=False,
         stateful=False,
         init='glorot_uniform',
         inner_init='orthogonal',
         forget_bias_init='one',
         activation='tanh',
         inner_activation='hard_sigmoid',
         W_regularizer=None,
         U_regularizer=None,
         b_regularizer=None,
         dropout_W=0.0,
         dropout_U=0.0,
         name="lstm_3"),
    Dense(256, name="dense_2"),
    Activation('tanh', name="act_2"),
    Dense(128, name="dense_3"),
    Activation('tanh', name="act_3"),
    Dense(64, name="dense_4"),
    Activation('tanh', name="act_4"),
    Dense(1, name="output")
])

model.compile(optimizer='adadelta',
              loss=rmse,
              metrics=['mae', 'mse'])

if os.path.isfile(model_weight_file):
    model.load_weights(model_weight_file, by_name=True)

plt.pause(0.5)
padding_left = 48
padding_right = 48

for i, row in df.iterrows():
    sep_pos = i

    if sep_pos > df.shape[0]:
        break

    # 开始准备要用于预测的数据
    if sep_pos >= (batch_size + timesteps):
        batch_dataset = np.zeros((batch_size + timesteps, timesteps, len(features)))
        df_start_pos = sep_pos - batch_size - timesteps
        batch_i = 0
        for time, row_ in df[df_start_pos:sep_pos].iterrows():
            # print(sep_pos, df_start_pos, batch_i)
            if batch_i >= timesteps:
                for t in range(timesteps):
                    step = timesteps - t
                    rec_index = df_start_pos + batch_i - step
                    # print("sep_pos:{} rec_index:{} df_start_pos:{} total:{} batch_i:{} step:{}".format(
                    #     sep_pos, rec_index, df_start_pos, df.shape[0], batch_i, step
                    # ))
                    # exit(0)
                    rec = df.iloc[rec_index, :]
                    batch_dataset[batch_i, t] = rec[features].values
                    # 如果超出了 则需要补位
            batch_i += 1
        batch_dataset = batch_dataset[timesteps:]

        pred_data = model.predict(batch_dataset, batch_size=batch_size)
        past_pred_line[0].set_data(np.arange(sep_pos + 1 - batch_size + prediction_step,
                                             sep_pos + 1),
                                   pred_data[:(batch_size - prediction_step)])
        pred_line[0].set_data(np.arange(sep_pos,
                                        sep_pos + 1 + prediction_step),
                              pred_data[(batch_size - prediction_step - 1):])

    sep_line[0].set_data(np.repeat(sep_pos, 2), (sep_max, sep_min))
    label = "batch id: {}\n" \
            "close price: {} " \
            "\n{}".format(sep_pos,
                          df[col][sep_pos],
                          df['time'][sep_pos])
    sep_text.set_text(label)
    sep_text.set_position((sep_pos + 20, sep_max - (sep_max - sep_min) * 0.15))

    past_line[0].set_data(np.arange(sep_pos + 1), [df[col].values[:sep_pos + 1]])

    ax.set_xlim(sep_pos - padding_left, sep_pos + padding_right)
    plt.pause(0.1)

plt.ioff()
plt.show()
exit()
