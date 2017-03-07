from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Convolution2D, BatchNormalization, Merge, LSTM, GRU, Input
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.regularizers import l1l2, activity_l1l2
import pandas as pd
import numpy as np
import Common.config as config
import os,time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from DataTransform.Transform5M import Transform5M as t5m


class ModelMACD:
    def __init__(self):
        name = "Model_MACD"
        self._model_file = os.path.join(config.MODEL_DIR, name + '_model.h5')
        self._weight_file = os.path.join(config.MODEL_DIR, name + '_weight.h5')

        self._encoder_file = os.path.join(config.MODEL_DIR, name + '_encoder.h5')
        self._encoder_weight_file = os.path.join(config.MODEL_DIR, name + '_encoder_weight.h5')

        input_data = Input(shape=(144,))
        encoded = Dense(80, activation='relu', name="encoder_1")(input_data)
        encoded = Dense(40, activation='relu', name="encoder_2")(encoded)
        encoded = Dense(20, activation='relu', name="encoder_3")(encoded)

        encoder_output = Dense(2)(encoded)
        decoded = Dense(20, activation='relu', name="decoder_1")(encoder_output)
        decoded = Dense(40, activation='relu', name="decoder_2")(decoded)
        decoded = Dense(80, activation='relu', name="decoder_3")(decoded)
        decoded = Dense(144, activation='relu', name="output")(decoded)

        self._model = Model(input=input_data, output=decoded)
        self.encoder = Model(input=input_data, output=encoder_output)

        print("Network output layout")
        for layer in self._model.layers:
            print(layer.output_shape)
        print("\n\n")
        #
        # rmsprop = RMSprop(lr=1e-9, rho=0.7, epsilon=1e-4, decay=1e-10)
        # sgd = SGD(lr=1e-6, decay=1e-7, momentum=0.5, nesterov=True)

        import keras.backend as K
        def mme(y_true, y_pred):
            return (K.mean(K.abs(y_pred - y_true))) / K.max(y_true)

        self._model.compile(optimizer='adadelta',  # adadelta
                            loss='mse',  # 'binary_crossentropy',
                            metrics=['mae',mme])

        if os.path.isfile(self._weight_file):
            self._model.load_weights(self._weight_file, by_name=True)

        return

    def _transform_inputs(self, input):
        # input = input.reshape(input.shape[0],-1)
        #
        # input = input.reshape(input.shape[0], 1, input.shape[1], input.shape[2])
        #
        price_vec_in = input[:, :, [0, 1, 2, 3]]
        price_change_in = input[:, :, [4, 5, 6, 7]]
        ma_in = input[:, :, [8, 9, 10, 11]]
        ema_in = input[:, :, [12, 13, 14, 15]]
        boll_in = input[:, :, [16, 17, 18]]
        vol_in = input[:, :, [19, 20, 21, 22, 23, 24, 25, 26]]
        cci_in = input[:, :, [27, 28, 29]]
        rsi_in = input[:, :, [30, 31, 32]]
        kdj_in = input[:, :, [33, 34, 35]]
        bias_in = input[:, :, [36, 37, 38]]
        roc_in = input[:, :, [39, 40]]
        change_in = input[:, :, [41]]
        amp_in = input[:, :, [42, 43, 44]]
        wr_in = input[:, :, [45, 46, 47]]
        mi_in = input[:, :, [48, 49, 50, 51]]
        oscv_in = input[:, :, [52]]
        dma_in = input[:, :, [53, 54]]
        abr_in = input[:, :, [55, 56]]
        mdi_in = input[:, :, [57, 58, 59, 60]]
        asi_in = input[:, :, [61, 62, 63, 64]]
        macd_in = input[:, :, [65, 66, 67]]
        psy_in = input[:, :, [68, 69]]
        emv_in = input[:, :, [70, 71]]
        wvad_in = input[:, :, [72, 73]]
        # #
        # # input = [
        # #     price_vec_in, price_change_in, ma_in, ema_in,
        # #     boll_in, vol_in, cci_in, rsi_in, kdj_in, bias_in, roc_in,
        # #     change_in, amp_in, wr_in, mi_in, oscv_in, dma_in, abr_in,
        # #     mdi_in, asi_in, macd_in, psy_in, emv_in, wvad_in
        # # ]
        # #
        input = [
            macd_in
        ]
        # 完全没有特征的
        # cci_in price_vec_in, price_change_in emv_in psy_in mdi_in rsi_in
        # 特征完全混在一起的
        # amp_in，
        # 已测试就梯度弥散
        # asi_in dma_in  abr_in wvad_in
        input = np.concatenate(input, axis=2)
        print(input.shape)
        input = input.reshape(input.shape[0], -1)

        v_max = 5
        v_min = -5
        input = ((input - v_min) / (v_max - v_min) + 1.5) ** 12
        return input


    def train(self, training_set, validation_set, test_set):
        batch_size = 32

        X_train, X_validation, X_test = training_set[0], validation_set[0], test_set[0]
        y_train, y_validation, y_test = training_set[1], validation_set[1], test_set[1]

        X_train = self._transform_inputs(X_train)
        X_validation = self._transform_inputs(X_validation)
        X_test = self._transform_inputs(X_test)

        y_test = self.scale_result(y_test)

        data = self.encoder.predict(X_test)

        print('Training ------------')
        value_dropout_count = 100
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(hspace=0.001,
                            wspace=0.01,
                            left=0.08,
                            right=1,
                            top=0.96,
                            bottom=0.06)
        ax.set_xlim(np.sort(data[:, 0])[(value_dropout_count - 1)],
                    np.sort(data[:, 0])[(-value_dropout_count)])
        ax.set_ylim(np.sort(data[:, 1])[(value_dropout_count - 1)],
                    np.sort(data[:, 1])[(-value_dropout_count)])
        s = ax.scatter(x=data[:, 0], y=data[:, 1], c=y_test, marker='.', cmap=plt.cm.jet)
        plt.colorbar(s)

        def animate(i):
            self._model.fit(X_train, X_train,
                            nb_epoch=20,
                            batch_size=64,
                            verbose=1,
                            validation_data=(X_validation, X_validation),
                            shuffle=True
                            )

            print('\nTesting ------------')
            loss = self._model.evaluate(X_test, X_test, batch_size=batch_size)
            print('\n\nloss: ', loss[0])
            print('mae: ', loss[1])
            print('mme: ', loss[2])
            print('--'*10)

            if i % 5 == 0:
                self._model.save_weights(self._weight_file)
                print("model saved\n\n")
                print('--' * 10)

            data = self.encoder.predict(X_validation)

            s.set_offsets(data)
            ax.set_xlim(np.sort(data[:, 0])[(value_dropout_count - 1)],
                        np.sort(data[:, 0])[(-value_dropout_count)])
            ax.set_ylim(np.sort(data[:, 1])[(value_dropout_count - 1)],
                        np.sort(data[:, 1])[(-value_dropout_count)])
            # ax.set_xlim(np.sort(data[:, 0])[(value_dropout_count - 1)],
            #             8000)
            # ax.set_ylim(np.sort(data[:, 1])[(value_dropout_count - 1)],
            #             6000)
            time.sleep(0.5)
            return s,

        ani = animation.FuncAnimation(fig, animate, np.arange(1, 2000))

        plt.show()

    def scale_result(self, dataset):
        # dataset[:, 0] = (dataset[:, 0] - self.result_min) / (self.result_max - self.result_min)
        # dataset[:, 0] *= 10
        return dataset
        # return new_dataset

    def rev_scale_result(self, dataset):

        # dataset[:, 0] *= 0.1
        # dataset[:, 0] = dataset[:, 0] * (self.result_max - self.result_min) + self.result_min
        return dataset

    def predict(self, data_set):
        X = self._transform_inputs(data_set)
        result = self._model.predict(X, verbose=0)
        # result = self.rev_scale_result(result)
        return result
