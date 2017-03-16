#!/usr/bin/env python3

import os, sys, datetime, keras, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

import Common.config as config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Convolution2D, BatchNormalization, Merge, LSTM, GRU
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
from Common.KerasMetrics import max_abs_error, mean_signed_deviation, root_mean_squared_error
from sklearn import preprocessing

np.random.seed(7)
np.set_printoptions(threshold=np.inf, linewidth=1000)

# 参数设置
# 对于 5步的 lookback 120 - 150 最好
# 对于 4步的 lookback 120 - 150 最好
prediction_step = 5
lookback_days = 100
training_size = 200

init_start_pos = 2
zoom_window_size = 15

training_epoch = 50
batch_size = 100

model_weight_file = os.path.join(config.MODEL_DIR, 'research_hist_step_{}_weight.h5')
data_file = os.path.join(config.CACHE_DIR, 'dataset.csv')


def create_model(input_dim, prediction_step):
    file = model_weight_file.format(prediction_step + 1)

    model = Sequential([
        Dense(200, input_dim=input_dim),
        Activation('relu'),
        BatchNormalization(),
        Dense(200),
        Activation('relu'),
        BatchNormalization(),
        Dense(200),
        Activation('relu'),
        BatchNormalization(),
        Dense(200),
        Activation('relu'),
        BatchNormalization(),
        Dense(200),
        Activation('relu'),
        BatchNormalization(),
        Dense(200),
        Activation('relu'),
        BatchNormalization(),
        Dense(1),
        Activation('linear'),
    ])

    model.compile(optimizer='adadelta',
                  loss=root_mean_squared_error,
                  metrics=[max_abs_error, 'mae', 'mse', mean_signed_deviation])

    if os.path.isfile(file):
        model.load_weights(file, by_name=True)

    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                  factor=0.2,
                                  verbose=1,
                                  patience=5,
                                  min_lr=0.001)
    checkpoint = ModelCheckpoint(file,
                                 monitor='loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='min',
                                 period=1)

    return {'model': model, 'callbacks': [reduce_lr, checkpoint]}


def init_graph(data):
    scaler = data.get('scaler')
    values = data.get('dataset')
    result = data.get('training_result')

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(3, 5)

    # 历史预览图
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.set_title('History Fitting - Perceptron #1')
    ax1.grid()
    ax1.plot(range(len(result)),
             result,
             color='b', alpha=0.8,
             marker='.')
    pred_tr_line = ax1.plot([0], result[0], color='g', alpha=0.8, marker='.')
    ax1.autoscale()

    # 预测图
    ax2 = fig.add_subplot(gs[0:2, 2:5])
    ax2.set_title('Future Prediction')
    act_future = data.get('test_result_act')

    ax2.plot(range(zoom_window_size),
             scaler.inverse_transform(values[-(zoom_window_size):]), color='b', alpha=0.9, marker='.')
    ax2.plot(range(zoom_window_size - 1, zoom_window_size + len(act_future)),
             list(scaler.inverse_transform(values[-1])) + list(act_future)
             , color='deepskyblue', alpha=1, marker='.')
    ax2.set_ylim(
        scaler.inverse_transform(
            [np.min(values[-(zoom_window_size):, 0].tolist() + list(scaler.transform(act_future)))]
        )[0] * 0.97,
        scaler.inverse_transform(
            [np.max(values[-(zoom_window_size):, 0].tolist() + list(scaler.transform(act_future)))]
        )[0] * 1.02)

    # 监控各感知器的拟合程度
    ax_s1 = fig.add_subplot(gs[2, 0])
    ax_s1.set_title('Perceptron - Step 1')
    ax_s2 = fig.add_subplot(gs[2, 1])
    ax_s2.set_title('Perceptron - Step 2')
    ax_s3 = fig.add_subplot(gs[2, 2])
    ax_s3.set_title('Perceptron - Step 3')
    ax_s4 = fig.add_subplot(gs[2, 3])
    ax_s4.set_title('Perceptron - Step 4')
    ax_s5 = fig.add_subplot(gs[2, 4])
    ax_s5.set_title('Perceptron - Step 5')

    ax_s1.grid()
    ax_s2.grid()
    ax_s3.grid()
    ax_s4.grid()
    ax_s5.grid()

    ax_s1.plot(range(zoom_window_size + prediction_step),
               scaler.inverse_transform(result[-(zoom_window_size + prediction_step):]), color='b', alpha=0.9,
               marker='.')

    ax_s2.plot(range(zoom_window_size + prediction_step),
               scaler.inverse_transform(result[-(zoom_window_size + prediction_step):]), color='b', alpha=0.9,
               marker='.')

    ax_s3.plot(range(zoom_window_size + prediction_step),
               scaler.inverse_transform(result[-(zoom_window_size + prediction_step):]), color='b', alpha=0.9,
               marker='.')

    ax_s4.plot(range(zoom_window_size + prediction_step),
               scaler.inverse_transform(result[-(zoom_window_size + prediction_step):]), color='b', alpha=0.9,
               marker='.')

    ax_s5.plot(range(zoom_window_size + prediction_step),
               scaler.inverse_transform(result[-(zoom_window_size + prediction_step):]), color='b', alpha=0.9,
               marker='.')

    fitting_step1_line = ax_s1.plot([0],
                                    scaler.inverse_transform(result[-(zoom_window_size + prediction_step)]), color='r',
                                    alpha=0.6,
                                    marker='.')
    fitting_step2_line = ax_s2.plot([0],
                                    scaler.inverse_transform(result[-(zoom_window_size + prediction_step)]), color='r',
                                    alpha=0.6,
                                    marker='.')
    fitting_step3_line = ax_s3.plot([0],
                                    scaler.inverse_transform(result[-(zoom_window_size + prediction_step)]), color='r',
                                    alpha=0.6,
                                    marker='.')
    fitting_step4_line = ax_s4.plot([0],
                                    scaler.inverse_transform(result[-(zoom_window_size + prediction_step)]), color='r',
                                    alpha=0.6,
                                    marker='.')
    fitting_step5_line = ax_s5.plot([0],
                                    scaler.inverse_transform(result[-(zoom_window_size + prediction_step)]), color='r',
                                    alpha=0.6,
                                    marker='.')

    pred_step1_line_zoom = ax2.plot([zoom_window_size - 1],
                                    scaler.inverse_transform(values[-1]), color='r', alpha=1,
                                    marker='o')
    pred_step2_line_zoom = ax2.plot([zoom_window_size - 1],
                                    scaler.inverse_transform(values[-1]), color='r', alpha=0.8,
                                    marker='o')
    pred_step3_line_zoom = ax2.plot([zoom_window_size - 1],
                                    scaler.inverse_transform(values[-1]), color='r', alpha=0.6,
                                    marker='o')
    pred_step4_line_zoom = ax2.plot([zoom_window_size - 1],
                                    scaler.inverse_transform(values[-1]), color='r', alpha=0.4,
                                    marker='o')
    pred_step5_line_zoom = ax2.plot([zoom_window_size - 1],
                                    scaler.inverse_transform(values[-1]), color='r', alpha=0.2,
                                    marker='o')

    ax2.grid()
    plt.tight_layout()

    plt.ion()
    plt.pause(0.5)

    return {
        'tr_line': pred_tr_line[0],
        'fitting_lines': [fitting_step1_line[0],
                          fitting_step2_line[0],
                          fitting_step3_line[0],
                          fitting_step4_line[0],
                          fitting_step5_line[0]],
        'pr_lines': [pred_step1_line_zoom[0],
                     pred_step2_line_zoom[0],
                     pred_step3_line_zoom[0],
                     pred_step4_line_zoom[0],
                     pred_step5_line_zoom[0]],
    }


def max_start_pos(peek_future=True):
    dataset = pd.read_csv(data_file, index_col=0)
    if peek_future:
        max_pos = dataset.shape[0] - prediction_step - lookback_days - training_size - prediction_step + 1
    else:
        max_pos = dataset.shape[0] - prediction_step - lookback_days - training_size + 1
    return max_pos


def prepare_data(start_pos, peek_future=True):
    dataset = pd.read_csv(data_file, index_col=0)

    # 从这里开始就看不见未来了
    start = start_pos
    end = start_pos + training_size + lookback_days + (prediction_step - 1)

    # 防止 start_pos 越界
    if end + prediction_step > dataset.shape[0]:
        if peek_future:
            shift = end + prediction_step - dataset.shape[0]
        else:
            shift = end - dataset.shape[0]
        start -= shift
        end -= shift

    # 先窥探一下未来
    if peek_future:
        act_future = dataset[end: end + prediction_step].values
    else:
        act_future = [None] * prediction_step

    dataset = dataset[start:end].values
    scaler = preprocessing.StandardScaler().fit(dataset)
    dataset = scaler.transform(dataset)

    # 假设每天都是盘后进行预测
    # 分别准备5个模型需要的训练集，结果集共享一个，都是最后到今日收盘的价格
    # 共享结果集

    training_set = []
    training_result = dataset[-training_size:].reshape(-1, 1)
    for i in range(prediction_step):
        training_set.append([])

    # 准备步长训练数据
    for i in range(prediction_step):
        step = prediction_step - i - 1
        for j in range(training_size):
            start = step + j
            end = start + lookback_days
            training_set[i].append(
                dataset[start:end]
            )
            # print(start, end, dataset.shape[0], dataset[start:end].shape[0])
        training_set[i] = np.vstack(training_set[i]).reshape(-1, lookback_days)

    # 准备测试数据 等于过去5组至今的回看数据
    # 因为最长视野的模型可从看到未来5步
    test_set = []
    test_result = scaler.transform(act_future)
    for i in range(prediction_step):
        test_set.append([])

    for i in range(prediction_step):
        step = prediction_step - i - 1
        for j in range(i + 1):
            start = training_size + step + j
            end = start + lookback_days
            test_set[i].append(
                dataset[start:end]
            )
        test_set[i] = np.vstack(test_set[i]).reshape(-1, lookback_days)

    return {
        'dataset': dataset,
        'test_set': test_set,
        'test_result_act': act_future,
        'test_result': test_result,
        'training_result': training_result,
        'training_set': training_set,
        'scaler': scaler
    }


# create models
models = [None, None, None, None, None]
for i in range(prediction_step):
    models[i] = create_model(lookback_days, i)

# init data
max_pos = max_start_pos(peek_future=True)
data = prepare_data(init_start_pos, peek_future=True)

# init graph
lines = init_graph(data)

# return {
#     'tr_line': pred_tr_line[0],
#     'fitting_lines': [fitting_step1_line[0],
#                       fitting_step2_line[0],
#                       fitting_step3_line[0],
#                       fitting_step4_line[0],
#                       fitting_step5_line[0]],
#     'pr_lines': [pred_step1_line_zoom[0],
#                  pred_step2_line_zoom[0],
#                  pred_step3_line_zoom[0],
#                  pred_step4_line_zoom[0],
#                  pred_step5_line_zoom[0]],
# }

# train models
for _ in range(100):
    for i in range(prediction_step):
        model = models[i].get('model')
        callbacks = models[i].get('callbacks')
        model.fit(data.get('training_set')[i], data.get('training_result'),
                  nb_epoch=training_epoch,
                  batch_size=batch_size,
                  callbacks=callbacks,
                  verbose=2)
        pred_train_y = model.predict(data.get('training_set')[i])

        # visualized training result
        if i == 0:
            lines.get('tr_line').set_data(range(len(pred_train_y)), pred_train_y)

        lines.get('fitting_lines')[i].set_data(range(zoom_window_size + prediction_step),
                                               data.get('scaler').inverse_transform(
                                                   pred_train_y[-(zoom_window_size + prediction_step):]))

    for i in range(prediction_step):
        pred_value = models[i].get('model').predict(data.get('test_set')[i])
        pred_value = data.get('scaler').inverse_transform(list(data.get('dataset')[-1]) + list(pred_value))
        lines.get('pr_lines')[i].set_data(range(zoom_window_size - 1, zoom_window_size + len(pred_value) - 1),
                                          pred_value)
        print(pred_value)
        print(i, '-' * 100)
    # 打印输出真实数据作比较
    print(data.get('scaler').inverse_transform(list(data.get('dataset')[-1]) + list(data.get('test_result_act'))))
    print(i, '-' * 100)

    plt.pause(0.5)

time.sleep(1000)
