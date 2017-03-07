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


class ModelPrice:
    def __init__(self):
        name = "Model_Price"
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
                            metrics=['mae', mme])

        if os.path.isfile(self._weight_file):
            self._model.load_weights(self._weight_file, by_name=True)

        return

    def _transform_inputs(self, input):
        price_vec_in = input[:, :, [0, 1, 2, 3]]
        price_change_in = input[:, :, [4, 5, 6, 7]]
        input = [
            price_vec_in, price_change_in
        ]
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
        value_dropout_count = 10
        fig, ax = plt.subplots()
        ax.set_xlim(np.sort(data[:, 0])[(value_dropout_count - 1)],
                    np.sort(data[:, 0])[(-value_dropout_count)])
        ax.set_ylim(np.sort(data[:, 1])[(value_dropout_count - 1)],
                    np.sort(data[:, 1])[(-value_dropout_count)])
        s = ax.scatter(x=data[:, 0], y=data[:, 1], c=y_test, cmap=plt.cm.jet)
        plt.colorbar(s)

        def animate(i):
            self._model.fit(X_train, X_train,
                            nb_epoch=10,
                            batch_size=256,
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
            time.sleep(0.5)
            return s,

        ani = animation.FuncAnimation(fig, animate, np.arange(1, 200))

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
