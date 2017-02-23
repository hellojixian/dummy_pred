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
from keras.layers import Dense, Activation, Flatten, Dropout, Convolution2D, BatchNormalization
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

        self._model = Sequential([
            Convolution2D(16, 15, 36, border_mode='same', input_shape=(1, 36, 36), dim_ordering ='th'),
            Activation('tanh'),
            # Dropout(0.5),
            # Convolution2D(32, 10, 36, border_mode='same'),
            # Activation('tanh'),
            # Dropout(0.25),
            Convolution2D(32, 10, 8, border_mode='same'),
            Activation('tanh'),
            Dropout(0.25),
            Convolution2D(32, 5, 10, border_mode='same'),
            Activation('tanh'),
            Dropout(0.25),
            Flatten(),
            Dense(1024),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Dense(512),
            # BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            # Dense(256),
            # BatchNormalization(),
            # Activation('relu'),
            # Dense(64),
            # BatchNormalization(),
            # Activation('relu'),
            # BatchNormalization(),
            Dense(32),
            # BatchNormalization(),
            Activation('relu'),
            Dense(3),
            Activation('softmax'),
        ])
        # self._model = Sequential([
        #     Dense(2048, input_dim=648),
        #     Activation('relu'),
        #     # Dropout(0.25),
        #     Dense(4096),
        #     Dense(4096),
        #     Activation('relu'),
        #     # Dropout(0.25),
        #     Dense(512),
        #     Activation('relu'),
        #     # Dropout(0.25),
        #     Dense(256),
        #     Activation('relu'),
        #     Dense(64),
        #     Activation('relu'),
        #     Dense(32),
        #     Activation('relu'),
        #     Dense(3),
        #     Activation('softmax'),
        # ])
        try:
            self._model = load_model(name)
        except Exception:
            pass

        rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)

        # We add metrics to get more results you want to see
        # sgd 曾经逼近到 89%
        self._model.compile(optimizer='adadelta',  # adadelta
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

        return

    def train(self, training_set, validation_set, test_set):

        X_train, X_validation, X_test = training_set[0], validation_set[0], test_set[0]
        y_train, y_validation, y_test = training_set[1], validation_set[1], test_set[1]

        # extra data transformation for fit the model
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
        X_validation = X_validation.reshape(X_validation.shape[0], 1, X_validation.shape[1], X_validation.shape[2])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

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
        while loss >= 0.0001: # 0.02:
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
            self._model.save(self._name)
            retry += 1
            if retry > 20:
                break
        return loss, accuracy

    def predict(self, X):
        X = X.reshape(1, 1, X.shape[0], X.shape[1])
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

        if use_cache and cache.has_cached_data():
            print("Loading data from cache")
            d = cache.load_cached_data()
        else:
            print("Loading data from query")
            d = t5m.prepare_data(stock_code, start_date, end_date)
            d = t5m.feature_extraction(d)
            d = t5m.feature_select(d)
            d = t5m.feature_scaling(d)
            d = cache.cache_data(d)
        d = t5m.feature_reshaping(d)
        X = d[1]
        y = t5m.prepare_result(d[0], stock_code, start_date, end_date)
        return X, y
