'''
这个模型通过看一天的K下午2点之前的K线，来判断第二天开盘是高开还是低开,
或者是在-0.5% 到 +0.5% 之间震荡

训练方法总结，
用多只股票在历史数据上训练，
得到的模型用机器没见过的股票在历史数据的后2-3个月上复盘，如果正确率达到95%以上
证明模型适用于该股票，可以用此模型来预测这只股票未来1个月的走势


训练模型的校验集合 N只未见过的股票最后2个月的走势

测试思路 训练目标
0，通过看当天下午2点之前的盘势，判断明早是否会高开超过0.5% 
1，通过看当天下午2点之前的盘势，判断在接下来的 10，30 分钟价格哪个是最低点，决定买入时机 
2，通过看昨天一天的盘式，判断第二天可能出现出现的卖出点会在什么时候，例如开盘就扔，等30分钟后再扔，等60分钟再扔
3，确定大致要扔的时间，如果开盘就扔那就立即直接扔，如果等30分钟，那么就用训练看30个分钟线的模型来判断未来是否会走低，如果感觉要走低了 就扔掉，
4，确定大致的买入时间，跟踪每分钟变化，如果觉得可能要涨了就赶紧买入吧
'''

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Convolution2D, BatchNormalization, Merge
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras.models import load_model

from DataCache.CacheManager import CacheManager
import DataTransform.Transform5M_T1 as t5m
import numpy as np
import os


class Model5MT1:
    _name = None
    _model = None

    def __init__(self, name):
        name = os.path.join('Models', name)
        self._name = name

        change_dim = 6
        price_dim = 7
        vol_dim = 8
        index_dim = 2
        cci_dim = 3
        kdj_dim = 3
        rsi_dim = 3
        bias_dim = 3
        boll_dim = 3

        change_model = Sequential([
            Convolution2D(88, 5, change_dim,
                          border_mode='valid',
                          input_shape=(1, 36, change_dim),
                          dim_ordering='th',
                          name="change_cnn_conv_1"),
            BatchNormalization(),
            Activation('tanh'),
            Dropout(0.25),
            Convolution2D(32, 8, 1,
                          border_mode='valid',
                          dim_ordering='th',
                          name="change_cnn_conv_2"),
            BatchNormalization(),
            Activation('tanh'),
            Dropout(0.25),
            Convolution2D(16, 1, 1,
                          border_mode='valid',
                          dim_ordering='th',
                          name="change_cnn_conv_3"),
            BatchNormalization(),
            Activation('tanh'),
            Dropout(0.25),
            Flatten(),
        ])

        price_model = Sequential([
            Convolution2D(88, 5, price_dim,
                          border_mode='valid',
                          input_shape=(1, 36, price_dim),
                          dim_ordering='th',
                          name="price_cnn_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Convolution2D(32, 8, 1,
                          border_mode='valid',
                          dim_ordering='th',
                          name="price_cnn_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Convolution2D(16, 5, 1,
                          border_mode='valid',
                          dim_ordering='th',
                          name="price_cnn_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Flatten(),
        ])

        vol_model = Sequential([
            Convolution2D(88, 5, vol_dim,
                          border_mode='valid',
                          input_shape=(1, 36, vol_dim),
                          dim_ordering='th',
                          name="vol_cnn_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Convolution2D(32, 8, 1,
                          border_mode='valid',
                          dim_ordering='th',
                          name="vol_cnn_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Convolution2D(16, 5, 1,
                          border_mode='valid',
                          dim_ordering='th',
                          name="vol_cnn_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Flatten(),
        ])

        index_model = Sequential([
            Convolution2D(32, 5, index_dim,
                          border_mode='valid',
                          input_shape=(1, 36, index_dim),
                          dim_ordering='th',
                          name="index_cnn_conv_1"),
            BatchNormalization(),
            Activation('tanh'),
            Dropout(0.25),
            Convolution2D(16, 5, 1,
                          border_mode='valid',
                          dim_ordering='th',
                          name="index_cnn_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Flatten()
        ])

        cci_model = Sequential([
            Convolution2D(32, 5, cci_dim,
                          border_mode='valid',
                          input_shape=(1, 36, cci_dim),
                          dim_ordering='th',
                          name="cci_cnn_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Convolution2D(16, 8, 1,
                          border_mode='valid',
                          dim_ordering='th',
                          name="cci_cnn_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Flatten()
        ])

        rsi_model = Sequential([
            Convolution2D(32, 5, rsi_dim,
                          border_mode='valid',
                          input_shape=(1, 36, rsi_dim),
                          dim_ordering='th',
                          name="rsi_cnn_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Convolution2D(16, 8, 1,
                          border_mode='valid',
                          dim_ordering='th',
                          name="rsi_cnn_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Flatten()
        ])

        kdj_model = Sequential([
            Convolution2D(32, 5, kdj_dim,
                          border_mode='valid',
                          input_shape=(1, 36, kdj_dim),
                          dim_ordering='th',
                          name="kdj_cnn_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Convolution2D(16, 8, 1,
                          border_mode='valid',
                          dim_ordering='th',
                          name="kdj_cnn_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Flatten()
        ])

        bias_model = Sequential([
            Convolution2D(32, 5, bias_dim,
                          border_mode='valid',
                          input_shape=(1, 36, bias_dim),
                          dim_ordering='th',
                          name="boll_cnn_conv_1"),
            BatchNormalization(),
            Activation('tanh'),
            Dropout(0.25),
            Convolution2D(16, 8, 1,
                          border_mode='valid',
                          dim_ordering='th',
                          name="boll_cnn_conv_2"),
            BatchNormalization(),
            Activation('tanh'),
            Dropout(0.25),
            Flatten()
        ])

        boll_model = Sequential([
            Convolution2D(32, 5, boll_dim,
                          border_mode='valid',
                          input_shape=(1, 36, boll_dim),
                          dim_ordering='th',
                          name="boll_cnn_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Convolution2D(16, 8, 1,
                          border_mode='valid',
                          dim_ordering='th',
                          name="boll_cnn_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Flatten()
        ])

        self._model = Sequential([
            Merge([change_model, price_model, vol_model,
                   index_model, cci_model, rsi_model,
                   kdj_model, bias_model, boll_model],
                  mode='concat',
                  concat_axis=-1,
                  name="dnn_merge_1"),
            Dense(4096, name="dnn_dense_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.5),
            Dense(1024, name="dnn_dense_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.5),
            Dense(512, name="dnn_dense_4"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Dense(256, name="dnn_dense_5"),
            Dropout(0.25),
            BatchNormalization(),
            Activation('relu'),
            Dense(32, name="dnn_dense_6"),
            BatchNormalization(),
            Activation('relu'),
            Dense(3, name="dnn_dense_7"),
            Activation('softmax'),
        ])

        print("Network output layout")
        for layer in self._model.layers:
            print(layer.output_shape)
        print("\n\n")
        # exit(0)

        rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)

        # We add metrics to get more results you want to see
        # sgd 曾经逼近到 89%
        self._model.compile(optimizer='adadelta',  # adadelta
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        try:
            # self._model = load_model(name)
            self._model.load_weights(name + "-weights", by_name=True)
        except Exception:
            pass

        return

    def _transform_inputs(self, input):
        input = input.reshape(input.shape[0], 1, input.shape[1], input.shape[2])
        # 0: "open_change"
        # 1: "high_change",
        # 2: "low_change",
        # 3: "close_change",

        # 4: "close",
        # 5: "ma5",
        # 6: "ma15",
        # 7: "ma25",
        # 8: "ma40",
        #
        # 9: "vol",
        # 10:"vr",
        # 11:"v_ma5",
        # 12:"v_ma15",
        # 13:"v_ma25",
        # 14:"v_ma40",
        #
        # 15: "cci_5",
        # 16: "cci_15",
        # 17: "cci_30",
        # 18: "rsi_6",
        # 19: "rsi_12",
        # 20: "rsi_24",
        # 21: "k9",
        # 22: "d9",
        # 23: "j9",
        # 24: "bias_5",
        # 25: "bias_10",
        # 26: "bias_30",
        #
        # 27: "boll_up",
        # 28: "boll_md",
        # 29: "boll_dn",
        # 30: "roc_12",
        # 31: "roc_25",
        #
        # 32: "change",
        # 33: "amplitude",
        # 34: "count",
        # 35: "turnover"]]

        change_in = input[:, :, :, [0, 1, 2, 3, 32, 33]]
        price_in = input[:, :, :, [4, 5, 6, 7, 8, 27, 29]]
        vol_in = input[:, :, :, [9, 10, 11, 12, 13, 14, 34, 35]]
        index_in = input[:, :, :, [30, 31]]
        cci_in = input[:, :, :, [15, 16, 17]]
        rsi_in = input[:, :, :, [18, 19, 20]]
        kdj_in = input[:, :, :, [21, 22, 23]]
        bias_in = input[:, :, :, [24, 25, 26]]
        boll_in = input[:, :, :, [27, 28, 29]]

        input = [change_in, price_in, vol_in, index_in, cci_in, rsi_in, kdj_in, bias_in, boll_in]
        return input

    def train(self, training_set, validation_set, test_set):

        X_train, X_validation, X_test = training_set[0], validation_set[0], test_set[0]
        y_train, y_validation, y_test = training_set[1], validation_set[1], test_set[1]

        # extra data transformation for fit the model
        X_train = self._transform_inputs(X_train)
        X_validation = self._transform_inputs(X_validation)
        X_test = self._transform_inputs(X_test)

        # X_train = X_train.reshape(X_train.shape[0], -1)
        # X_validation = X_validation.reshape(X_validation.shape[0], -1)
        # X_test = X_test.reshape(X_test.shape[0], -1)

        callbacks = [
            # EarlyStopping(monitor='val_acc', min_delta=0.05, patience=3, verbose=1, mode='max')
            EarlyStopping(monitor='val_loss', min_delta=0.05, patience=3, verbose=1, mode='min')
        ]

        print('Training ------------')
        loss, accuracy = 100, 0
        # Another way to train the model
        retry = 0
        # while accuracy <= 0.98:
        while loss >= 0.05:  # 0.02:
            self._model.fit(X_train, y_train,
                            nb_epoch=2,
                            batch_size=100,
                            callbacks=callbacks,
                            verbose=1,
                            validation_data=(X_validation, y_validation),
                            shuffle=True
                            )

            print('\nTesting ------------')
            # Evaluate the model with the metrics we defined earlier
            loss, accuracy = self._model.evaluate(X_test, y_test, batch_size=200)
            print('\ntest accuracy: ', accuracy)
            print('test loss: ', loss)
            self._model.save(self._name + "-model")
            self._model.save_weights(self._name + "-weights")
            retry += 1
            if retry > 20:
                break
        return loss, accuracy

    def predict(self, X):
        X = self._transform_inputs(X)
        cls = self._model.predict_classes(X, verbose=0)
        proba = self._model.predict_proba(X, verbose=0)
        proba *= 100
        proba = np.round(proba, 0)
        # print(cls, proba)
        # r = r[0] * 10000
        # r = np.round(r).astype(dtype=np.int8)
        # print(r)
        return cls[0], proba[0]

    def prepare_data(self, stock_code, start_date, end_date, use_cache=True):
        TMP_DATA_TABLE_NAME = 'transformed_stock_trading_5min_t1_data_' + stock_code
        cache = CacheManager(TMP_DATA_TABLE_NAME)

        t5m.init(stock_code)
        if use_cache and cache.has_cached_data():
            print("Loading data from cache")
            d = cache.load_cached_data()
        else:
            print("Loading data from query")
            d = t5m.prepare_data(start_date, end_date)
            d = t5m.feature_extraction(d)
            d = t5m.feature_select(d)
            d = cache.cache_data(d)
        d = t5m.feature_scaling(d)
        d = t5m.feature_reshaping(d)
        X = d[1]
        y = t5m.prepare_result(d[0], stock_code, start_date, end_date)
        return X, y
