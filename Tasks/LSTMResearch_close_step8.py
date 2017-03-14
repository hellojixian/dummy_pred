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

np.set_printoptions(threshold=np.inf, linewidth=1000)

# 参数设置
start_date = datetime.date(2016, 1, 1)
end_date = datetime.date(2016, 12, 30)
code = 'sz002166'
col = 'change'
data_splitter = 0.8
test_splitter = 0.5

features = ['change']

# features = ['ma25']
timesteps = 48  # 过去两个小时的走势
prediction_step = 1  #
batch_size = 20  # 一天有48个五分钟

model_weight_file = os.path.join(config.MODEL_DIR, 'LSTMResearchNextCloseS8_weight.h5')
ds_cache_file = os.path.join(config.CACHE_DIR, 'LSTMResearchDataset-{}-{}-s8.csv'.format(col, code))
rs_cache_file = os.path.join(config.CACHE_DIR, 'LSTMResearchResult-{}-{}-s8.csv'.format(col, code))
raw_cache_file = os.path.join(config.CACHE_DIR, 'LSTMResearchRaw-{}-{}-s8.csv'.format(col, code))

np.random.seed(7)
if os.path.isfile(ds_cache_file):
    ds_df = pd.read_csv(ds_cache_file)
    dataset = ds_df.values
    dataset = dataset.reshape(dataset.shape[0], timesteps, len(features))

    rs_df = pd.read_csv(rs_cache_file)
    result = rs_df

    df = pd.read_csv(raw_cache_file)
else:
    # 获取数据
    session = sessionmaker()
    session.configure(bind=config.DB_CONN)
    db = session()

    columns_list = ['time'] + features
    columns_str = "`time`, `change` "
    # columns_str = "`time`,`ma25`"
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
    # df.columns = ['close', 'ma5', 'ma15', 'ma25', 'ma40']
    db.close()

    # exit(0)
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
    shift_size = dataset.shape[0] - prediction_step - timesteps
    # df = df[:shift_size]
    dataset = dataset[:shift_size]
    result = result[:shift_size]

    # 缓存
    ds_df = pd.DataFrame(dataset.reshape(dataset.shape[0], -1))
    ds_df.to_csv(path_or_buf=ds_cache_file, index=False)

    rs_df = result
    rs_df.to_csv(path_or_buf=rs_cache_file, index=False)

    df.to_csv(path_or_buf=raw_cache_file, index=False)
# dataset = dataset[:500]
# result = result[:500]
# print(dataset[4:6])
# print(result[4:6])
# exit()

scaler_min = -0.05
scaler_max = 0.05

# 把输入输出都转成分类问题
from keras.utils.np_utils import to_categorical

input_features = 400
categories = 20

dataset = np.tanh((dataset - scaler_min) / (scaler_max-scaler_min)-0.5) * input_features
dataset_new = np.zeros((dataset.shape[0], timesteps, input_features))

# print(dataset_new[:20])
for i in range(dataset.shape[0]):
    for j in range(timesteps):
        dataset_new[i, j] = to_categorical(dataset[i, j, 0], input_features)
# print(dataset_new[:2])
dataset = dataset_new
result['change'] = (result['change'] * 10 + 1) * 10
result = result.values
result = to_categorical(result, categories)
print(result[:20])


print("dataset: {}\t result:{}".format(dataset.shape[0], result.shape[0]))
total = int(dataset.shape[0] / batch_size) * batch_size
dataset = dataset[:total]
result = result[:total]

sep_pt = int(dataset.shape[0] * data_splitter)
sep_pt = int(int((sep_pt / batch_size)) * batch_size)

sep_pt2 = (dataset.shape[0] - sep_pt) * test_splitter
sep_pt2 = int(int((sep_pt2 / batch_size)) * batch_size)
#
# print("after dropout:{}\t ".format(total))
# print("sep_pt: {}".format(sep_pt))
# print("sep_pt2: {}".format(sep_pt2))

# 整理训练数据集
training_X = dataset[:sep_pt]
validation_X = dataset[sep_pt:]
test_X = validation_X[sep_pt2:]
validation_X = validation_X[:sep_pt2]

result_arr = result
training_y = result_arr[:sep_pt]
validation_y = result_arr[sep_pt:]
test_y = validation_y[sep_pt2:]
validation_y = validation_y[:sep_pt2]
print(training_X.shape, validation_X.shape, test_X.shape)
print(training_y.shape, validation_y.shape, test_y.shape)

# 准备训练
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              verbose=1,
                              patience=5,
                              min_lr=0.001)
checkpoint = ModelCheckpoint(model_weight_file,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min',
                             period=1)
model = Sequential([
    # LSTM(50,
    #      input_shape=(timesteps, input_features),
    #      return_sequences=False,
    #      stateful=False,
    #      init='glorot_uniform',
    #      inner_init='orthogonal',
    #      forget_bias_init='one',
    #      activation='relu',
    #      inner_activation='hard_sigmoid',
    #      W_regularizer=None,
    #      U_regularizer=None,
    #      b_regularizer=None,
    #      dropout_W=0.0,
    #      dropout_U=0.0,
    #      name="lstm_1"),
    # LSTM(30,
    #      input_shape=(timesteps, input_features),
    #      return_sequences=False,
    #      stateful=False,
    #      init='glorot_uniform',
    #      inner_init='orthogonal',
    #      forget_bias_init='one',
    #      activation='relu',
    #      inner_activation='hard_sigmoid',
    #      W_regularizer=None,
    #      U_regularizer=None,
    #      b_regularizer=None,
    #      dropout_W=0.0,
    #      dropout_U=0.0,
    #      name="lstm_2"),
    #
    Dense(256, name="dense_2", input_dim=(input_features * timesteps)),
    Activation('relu', name="act_2"),
    Dropout(0.5),
    Dense(128, name="dense_3"),
    Activation('relu', name="act_3"),
    Dropout(0.5),
    Dense(64, name="dense_4"),
    Activation('relu', name="act_4"),
    Dropout(0.5),
    Dense(categories),
    Activation('softmax')
])
training_X = training_X.reshape(training_X.shape[0], -1)
validation_X = validation_X.reshape(validation_X.shape[0], -1)
print(training_X.shape)

model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

if os.path.isfile(model_weight_file):
    model.load_weights(model_weight_file, by_name=True)

# training_pred = model.predict(training_X, batch_size=batch_size)
# training_line[0].set_data(np.arange(0,
#                                     len(training_pred)), training_pred)
#
# validation_pred = model.predict(validation_X, batch_size=batch_size)
# validation_line[0].set_data(np.arange(sep_pt,
#                                       sep_pt + len(validation_pred)), validation_pred)
#
# test_pred = model.predict(test_X, batch_size=batch_size)
# test_line[0].set_data(np.arange(sep_pt + sep_pt2,
#                                 sep_pt + sep_pt2 + len(test_pred)), test_pred)
#
# plt.pause(0.5)
# plt.ioff()
# plt.show()
# exit()
model.fit(training_X, training_y,
          nb_epoch=2000,
          batch_size=batch_size,
          callbacks=[checkpoint, reduce_lr],
          verbose=1,
          validation_data=(validation_X, validation_y))
