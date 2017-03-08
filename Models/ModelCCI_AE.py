from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Convolution2D, BatchNormalization, Merge, LSTM, GRU, Input
import numpy as np
import Common.config as config
import os, math
from Common.KerasCallbacks import DataVisualized, DataTester
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


class ModelCCI:
    def __init__(self):
        name = "Model_CCI"
        self._model_file = os.path.join(config.MODEL_DIR, name + '_model.h5')
        self._model_weight_file = os.path.join(config.MODEL_DIR, name + '_weight.h5')

        encoded_dim = 2
        data_dim = 144

        input_data = Input(shape=(data_dim,))
        encoded = Dense(120, activation='relu', name="encoder_2")(input_data)
        encoded = Dense(80, activation='relu', name="encoder_3")(encoded)
        encoded = Dense(40, activation='relu', name="encoder_4")(encoded)
        encoder_output = Dense(encoded_dim, name="encoder_output")(encoded)

        decoded = Dense(40, activation='relu', name="decoder_1")(encoder_output)
        decoded = Dense(80, activation='relu', name="decoder_2")(decoded)
        decoded = Dense(120, activation='relu', name="decoder_3")(decoded)
        decoded = Dense(data_dim, activation='relu', name="decoder_output")(decoded)

        self._model = Model(input=input_data, output=decoded)

        decoder_input_data = Input(shape=(encoded_dim,))
        for layer in self._model.layers:
            if layer.name == 'decoder_1':
                decoder_output = layer(decoder_input_data)
                break

        self.encoder = Model(input=input_data, output=encoder_output)
        self.decoder = Model(input=decoder_input_data, output=decoder_output)

        print("Network output layout")
        for layer in self._model.layers:
            print(layer.name, layer.output_shape)
        print("\n\n")

        from Common.KerasMetrics import mean_error_rate
        self._model.compile(optimizer='adadelta',  # adadelta
                            loss='mse',
                            metrics=['mae', mean_error_rate])

        if os.path.isfile(self._model_weight_file):
            self._model.load_weights(self._model_weight_file, by_name=True)

        return

    def _transform_inputs(self, input):

        cci_in = input[:, :,  [27, 28, 29]]
        input = [
            cci_in
        ]
        input = np.concatenate(input, axis=2)
        print(input.shape)

        v_max = 3
        v_min = -3

        print("\nraw input range: {} to {}".format(np.min(input), np.max(input)))
        print("adjusted range limit: {} to {}".format(v_min, v_max))

        input = ((input - v_min) / (v_max - v_min)) - 0.5
        print(input.shape)
        # input = input.reshape(-1)
        input = np.tanh(input)
        input += 2
        input = input ** 10
        input = input.reshape(input.shape[0], -1)


        # input = ((input - v_min) / (v_max - v_min) + 1.5) ** 12
        print("adjusted input range: {} to {}".format(np.min(input), np.max(input)))
        print("transformed input shape: ", input.shape)
        print("--" * 20)
        # exit(0)
        return input

    def train(self, training_set, validation_set, test_set):
        X_train, X_validation, X_test = training_set[0], validation_set[0], test_set[0]
        y_train, y_validation, y_test = training_set[1], validation_set[1], test_set[1]

        X_train = self._transform_inputs(X_train)
        X_validation = self._transform_inputs(X_validation)
        X_test = self._transform_inputs(X_test)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.2,
                                      verbose=1,
                                      patience=5,
                                      min_lr=0.001)
        checkpoint = ModelCheckpoint(self._model_weight_file,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='min',
                                     period=1)
        data_vis = DataVisualized(self.encoder,
                                  X_validation,
                                  y_validation,
                                  value_dropout_count=100)
        tester = DataTester(X_test, period=5)

        self._model.fit(X_train, X_train,
                        nb_epoch=10000,
                        batch_size=256,
                        callbacks=[data_vis, reduce_lr, checkpoint, tester],
                        validation_data=(X_validation, X_validation),
                        shuffle=True)

    def predict(self, data_set):
        X = self._transform_inputs(data_set)
        result = self._model.predict(X, verbose=0)
        return result
