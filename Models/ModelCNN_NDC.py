from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Convolution2D, BatchNormalization, Merge
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras.models import load_model
import pandas as pd
import numpy as np
import os

from DataTransform.Transform5M import Transform5M as t5m


class Model_CNN_NDC:
    def __init__(self):
        self._name = "Model_CNN_NDC"

        price_vec_dim = 4
        price_change_dim = 4
        ma_dim = 4
        ema_dim = 4
        boll_dim = 3
        vol_dim = 8
        cci_dim = 3
        rsi_dim = 3
        kdj_dim = 3
        rsi_dim = 3
        bias_dim = 3
        roc_dim = 2
        change_dim = 1
        amp_dim = 3
        wr_dim = 3
        mi_dim = 4
        oscv_dim = 1
        dma_dim = 2
        abr_dim = 2
        mdi_dim = 4
        asi_dim = 4
        macd_dim = 3
        psy_dim = 2
        emv_dim = 2
        wvad_dim = 2

        change_model = Sequential([
            Convolution2D(88, 5, change_dim, border_mode='valid', input_shape=(1, 36, change_dim), dim_ordering='th'),
            BatchNormalization(),
            Activation('tanh'),
            Dropout(0.25),
            Convolution2D(32, 8, 1, border_mode='valid', dim_ordering='th'),
            BatchNormalization(),
            Activation('tanh'),
            Dropout(0.25),
            Convolution2D(16, 1, 1, border_mode='valid', dim_ordering='th'),
            BatchNormalization(),
            Activation('tanh'),
            Dropout(0.25),
            Flatten(),
        ])

        price_model = Sequential([
            Convolution2D(88, 5, price_dim, border_mode='valid', input_shape=(1, 36, price_dim), dim_ordering='th'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Convolution2D(32, 8, 1, border_mode='valid', dim_ordering='th'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Convolution2D(16, 5, 1, border_mode='valid', dim_ordering='th'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Flatten(),
        ])

        vol_model = Sequential([
            Convolution2D(88, 5, vol_dim, border_mode='valid', input_shape=(1, 36, vol_dim), dim_ordering='th'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Convolution2D(32, 8, 1, border_mode='valid', dim_ordering='th'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Convolution2D(16, 5, 1, border_mode='valid', dim_ordering='th'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Flatten(),
        ])

        index_model = Sequential([
            Convolution2D(32, 5, index_dim, border_mode='valid', input_shape=(1, 36, index_dim), dim_ordering='th'),
            BatchNormalization(),
            Activation('tanh'),
            Dropout(0.25),
            Convolution2D(16, 5, 1, border_mode='valid', dim_ordering='th'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Flatten()
        ])

        cci_model = Sequential([
            Convolution2D(32, 5, cci_dim, border_mode='valid', input_shape=(1, 36, cci_dim), dim_ordering='th'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Convolution2D(16, 8, 1, border_mode='valid', dim_ordering='th'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Flatten()
        ])

        rsi_model = Sequential([
            Convolution2D(32, 5, rsi_dim, border_mode='valid', input_shape=(1, 36, rsi_dim), dim_ordering='th'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Convolution2D(16, 8, 1, border_mode='valid', dim_ordering='th'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Flatten()
        ])

        kdj_model = Sequential([
            Convolution2D(32, 5, kdj_dim, border_mode='valid', input_shape=(1, 36, kdj_dim), dim_ordering='th'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Convolution2D(16, 8, 1, border_mode='valid', dim_ordering='th'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Flatten()
        ])

        bias_model = Sequential([
            Convolution2D(32, 5, bias_dim, border_mode='valid', input_shape=(1, 36, bias_dim), dim_ordering='th'),
            BatchNormalization(),
            Activation('tanh'),
            Dropout(0.25),
            Convolution2D(16, 8, 1, border_mode='valid', dim_ordering='th'),
            BatchNormalization(),
            Activation('tanh'),
            Dropout(0.25),
            Flatten()
        ])

        boll_model = Sequential([
            Convolution2D(32, 5, boll_dim, border_mode='valid', input_shape=(1, 36, boll_dim), dim_ordering='th'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Convolution2D(16, 8, 1, border_mode='valid', dim_ordering='th'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.25),
            Flatten()
        ])

        input_models = [
            price_vec_model, price_change_model, ma_model, ema_model,
            boll_model, vol_model, cci_model, rsi_model, kdj_model, bias_model, roc_model,
            change_model, amp_model, wr_model, mi_model, oscv_model, dma_model, abr_model,
            mdi_model, asi_model, macd_model, psy_model, emv_model, wvad_model
        ]
        self._model = Sequential([
            Merge(input_models, mode='concat',
                  concat_axis=-1),
            # Dense(4096),
            # BatchNormalization(),
            # Activation('linear'),
            # Dropout(0.5),
            Dense(1024, init='normal'),
            # BatchNormalization(),
            Activation('linear'),
            Dropout(0.5),
            Dense(512, init='normal'),
            # BatchNormalization(),
            Activation('linear'),
            Dropout(0.25),
            Dense(256, init='normal'),
            Dropout(0.25),
            # BatchNormalization(),
            Activation('linear'),
            Dense(32, init='normal'),
            # BatchNormalization(),
            Activation('linear'),
            Dense(1)
        ])

        print("Network output layout")
        for layer in self._model.layers:
            print(layer.output_shape)
        print("\n\n")
        exit(0)

        try:
            # self._model = load_model(self._name)
            self._model.load_weights(self._name + "-weights", by_name=True)
        except Exception:
            pass

        # rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)

        self._model.compile(optimizer='adadelta',  # adadelta
                            loss='mean_squared_error',
                            metrics=['mean_absolute_error'])
        return

    def data_features(self):
        features = t5m.features()
        for i in range(len(features)):
            print("{} - {}".format(i, features[i]))
        return

    def _transform_inputs(self, input):
        input = input.reshape(input.shape[0], 1, input.shape[1], input.shape[2])

        price_vec_in = input[:, :, :, [0, 1, 2, 3]]
        price_change_in = input[:, :, :, [4, 5, 6, 7]]
        ma_in = input[:, :, :, [8, 9, 10, 11]]
        ema_in = input[:, :, :, [12, 13, 14, 15]]
        boll_in = input[:, :, :, [16, 17, 18]]
        vol_in = input[:, :, :, [19, 20, 21, 22, 23, 24, 25, 26]]
        cci_in = input[:, :, :, [27, 28, 29]]
        rsi_in = input[:, :, :, [30, 31, 32]]
        kdj_in = input[:, :, :, [33, 34, 35]]
        bias_in = input[:, :, :, [36, 37, 38]]
        roc_in = input[:, :, :, [39, 40]]
        change_in = input[:, :, :, [41]]
        amp_in = input[:, :, :, [42, 43, 44]]
        wr_in = input[:, :, :, [45, 46, 47]]
        mi_in = input[:, :, :, [48, 49, 50, 51]]
        oscv_in = input[:, :, :, [52]]
        dma_in = input[:, :, :, [53, 54]]
        abr_in = input[:, :, :, [55, 56]]
        mdi_in = input[:, :, :, [57, 58, 59, 60]]
        asi_in = input[:, :, :, [61, 62, 63, 64]]
        macd_in = input[:, :, :, [65, 66, 67]]
        psy_in = input[:, :, :, [68, 69]]
        emv_in = input[:, :, :, [70, 71]]
        wvad_in = input[:, :, :, [72, 73]]

        input = [
            price_vec_in, price_change_in, ma_in, ema_in,
            boll_in, vol_in, cci_in, rsi_in, kdj_in, bias_in, roc_in,
            change_in,amp_in,wr_in,mi_in,oscv_in,dma_in,abr_in,
            mdi_in,asi_in,macd_in,psy_in,emv_in,wvad_in
        ]
        return input

    def train(self, training_set, validation_set, test_set):

        X_train, X_validation, X_test = training_set[0], validation_set[0], test_set[0]
        y_train, y_validation, y_test = training_set[1], validation_set[1], test_set[1]

        X_train = self._transform_inputs(X_train)
        X_validation = self._transform_inputs(X_validation)
        X_test = self._transform_inputs(X_test)

        callbacks = [
            EarlyStopping(monitor='val_loss', min_delta=0.05, patience=3, verbose=1, mode='min')
        ]

        print('Training ------------')
        loss, accuracy = 100, 0
        # Another way to train the model
        retry = 0
        # while accuracy <= 0.98:
        while loss >= 0.5:  # 0.02:
            self._model.fit(X_train, y_train,
                            nb_epoch=2,
                            batch_size=100,
                            callbacks=callbacks,
                            verbose=1,
                            validation_data=(X_validation, y_validation),
                            shuffle=True
                            )

            print('\nTesting ------------')
            loss, accuracy = self._model.evaluate(X_test, y_test, batch_size=200)
            print('\ntest loss: ', loss)
            print('test accuracy: ', accuracy)
            self._model.save(self._name)
            self._model.save_weights(self._name + "-weights")
            retry += 1
            if retry > 20:
                break
        return loss, accuracy

    def predict(self, data_set):
        X = self._transform_inputs(data_set)
        result = self._model.predict(X, verbose=0)
        return result
