from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Convolution2D, BatchNormalization, Merge, LSTM, GRU, Input
from Common.KerasCallbacks import DataVisualized, DataTester
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import Common.config as config
import os


class ModelMACD:
    def __init__(self):
        name = "Model_MACD"
        self._model_file = os.path.join(config.MODEL_DIR, name + '_model.h5')
        self._model_weight_file = os.path.join(config.MODEL_DIR, name + '_weight.h5')

        encoded_dim = 2
        data_dim = 144

        input_data = Input(shape=(data_dim,))
        encoded = Dense(80, activation='relu', name="encoder_1")(input_data)
        encoded = Dense(40, activation='relu', name="encoder_2")(encoded)
        encoded = Dense(20, activation='relu', name="encoder_3")(encoded)
        encoder_output = Dense(encoded_dim, name="dense_1")(encoded)

        decoded = Dense(20, activation='relu', name="decoder_1")(encoder_output)
        decoded = Dense(40, activation='relu', name="decoder_2")(decoded)
        decoded = Dense(80, activation='relu', name="decoder_3")(decoded)
        decoded = Dense(data_dim, activation='relu', name="output")(decoded)

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
