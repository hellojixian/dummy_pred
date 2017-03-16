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
from Common.KerasMetrics import max_abs_error,mean_signed_deviation,root_mean_squared_error
from sklearn import preprocessing

np.set_printoptions(threshold=np.inf, linewidth=1000)

# 参数设置
start_date = datetime.date(2016, 1, 1)
end_date = datetime.date(2016, 12, 30)

model_weight_file = os.path.join(config.MODEL_DIR, 'research_cos_weight.h5')

samples = 200
sep_pt = 80
sep_pt2 = 50

window_size = 80
features = 2
pred_step = 6


def gen_result(x):
    # return np.cos(x / 5)
    noise_1 = np.random.uniform(-0.2, 0.2)
    act_value = np.cos(x / 5) * np.sin(x/100 * np.sqrt(np.pi) + np.random.uniform(-0.2, 0.2)) + np.random.uniform(-0.2, 0.2)
    act_value *= np.cos(x/10)
    act_value = np.sqrt(1+np.e ** act_value) * 5 + 100
    # act_value += x ** 0.1
    return [noise_1,act_value]


def gen_dataset(x):
    return list(map(gen_result, np.arange(x - window_size, x)))


index = np.arange(0, samples)
dataset = list(map(gen_dataset, np.arange(0, samples)))
# # dataset = np.array(dataset).reshape(len(dataset), window_size)
# print((dataset[:1]))
# exit()
result = list(map(gen_result, np.arange(pred_step, samples + pred_step)))

dataset = np.array(dataset).reshape(len(dataset), window_size * features)
result = np.array(result).reshape(len(result), features)
result = result[:,1]

dataset = preprocessing.scale(dataset)
result = preprocessing.scale(result)

np.random.seed(7)

# 整理训练数据集
training_X = dataset[:sep_pt]
training_index = index[:sep_pt]
validation_X = dataset[sep_pt:sep_pt + sep_pt2]
validation_index = index[sep_pt:sep_pt + sep_pt2]
test_X = dataset[sep_pt + sep_pt2:]
test_index = index[sep_pt + sep_pt2:]

training_y = result[:sep_pt]
validation_y = result[sep_pt:sep_pt + sep_pt2]
test_y = result[sep_pt + sep_pt2:]

print(len(dataset), len(training_X), len(validation_X), len(test_X))
print(training_X)
print(validation_X)
print(test_X)
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
    Dense(200, input_dim=window_size*features),
    Activation('relu'),
    Dense(100),
    Activation('relu'),
    Dense(100),
    Activation('relu'),
    Dense(100),
    Activation('relu'),
    Dense(1),
    Activation('linear'),
])

model.compile(optimizer='adadelta',
              loss=root_mean_squared_error,
              metrics=[max_abs_error,'mae','mse',mean_signed_deviation])

if os.path.isfile(model_weight_file):
    model.load_weights(model_weight_file, by_name=True)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(training_index, training_y, color='b', alpha=0.4,  marker='.')
ax.plot(validation_index, validation_y, color='g', alpha=0.4, marker='.')
ax.plot(test_index, test_y, color='purple', alpha=0.4, marker='.')
pred_tr_line = ax.plot([0], training_y[0], color='g')
pred_val_line = ax.plot([0], training_y[0], color='y', alpha=0.8)
pred_test_line = ax.plot([0], training_y[0], color='pink', alpha=0.8)
plt.grid()
plt.tight_layout()
plt.ion()
plt.autoscale()
plt.pause(0.5)

# plt.ioff(); plt.show()


for _ in range(100):
    model.fit(training_X, training_y,
              nb_epoch=10,
              batch_size=32,
              callbacks=[
                  # checkpoint,
                  reduce_lr
              ],
              verbose=2,
              validation_data=(validation_X, validation_y))
    pred_train_y = model.predict(training_X)
    pred_val_y = model.predict(validation_X)
    pred_test_y = model.predict(test_X)
    pred_tr_line[0].set_data(training_index, pred_train_y)
    pred_val_line[0].set_data(validation_index, pred_val_y)
    pred_test_line[0].set_data(test_index, pred_test_y)
    plt.pause(0.5)
