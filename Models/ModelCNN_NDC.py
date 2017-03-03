from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Convolution2D, BatchNormalization, Merge
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras.models import load_model
import numpy as np
import os


class Model_CNN_NDC:
    def __init__(self):
        self._name = "Model_CNN_NDC"

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

        self._model = Sequential([
            Merge([change_model, price_model, vol_model,
                   index_model, cci_model, rsi_model,
                   kdj_model, bias_model, boll_model], mode='concat',
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
        # exit(0)

        try:
            #self._model = load_model(self._name)
            self._model.load_weights(self._name + "-weights", by_name=True)
        except Exception:
            pass

        # rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)

        self._model.compile(optimizer='adadelta',  # adadelta
                            loss='mean_squared_error',
                            metrics=['mean_absolute_error'])
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
