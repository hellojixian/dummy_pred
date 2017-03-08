from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Convolution2D, BatchNormalization, Merge
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.regularizers import l1l2, activity_l1l2
import pandas as pd
import numpy as np
from keras.utils import np_utils
import Common.config as config
import os
from DataTransform.Transform5M import Transform5M as t5m
from Models.ModelAMP_AE import ModelAMP
from Models.ModelASI_AE import ModelASI
from Models.ModelBIAS_AE import ModelBIAS
from Models.ModelBOLL_AE import ModelBOLL
from Models.ModelCCI_AE import ModelCCI
from Models.ModelChange_AE import ModelChange
from Models.ModelCount_AE import ModelCount
from Models.ModelEMA_AE import ModelEMA
from Models.ModelEMV_AE import ModelEMV
from Models.ModelKDJ_AE import ModelKDJ
from Models.ModelMACD_AE import ModelMACD
from Models.ModelMI_AE import ModelMI
from Models.ModelOSCV_AE import ModelOSCV
from Models.ModelPSY_AE import ModelPSY
from Models.ModelROC_AE import ModelROC
from Models.ModelTR_AE import ModelTR
from Models.ModelWVAD_AE import ModelWVAD
from Models.ModelRSI_AE import ModelRSI
from Models.ModelMDI_AE import ModelMDI


class ModelNDC_AE_DNN:
    def __init__(self, result_min, result_max, categories):
        name = "ModelNDC_AE_DNN"
        self._model_file = os.path.join(config.MODEL_DIR, name + '_model.h5')
        self._weight_file = os.path.join(config.MODEL_DIR, name + '_weight.h5')

        self.result_min = result_min
        self.result_max = result_max
        self.result_categores = categories

        model_amp = ModelAMP()
        enc_amp = model_amp.get_encoder()

        model_asi = ModelASI()
        enc_asi = model_asi.get_encoder()

        model_bias = ModelBIAS()
        enc_bias = model_bias.get_encoder()

        model_boll = ModelBOLL()
        enc_boll = model_boll.get_encoder()

        model_cci = ModelCCI()
        enc_cci = model_cci.get_encoder()

        model_change = ModelChange()
        enc_change = model_change.get_encoder()

        model_count = ModelCount()
        enc_count = model_count.get_encoder()

        model_ema = ModelEMA()
        enc_ema = model_ema.get_encoder()

        model_emv = ModelEMV()
        enc_emv = model_emv.get_encoder()

        model_kdj = ModelKDJ()
        enc_kdj = model_kdj.get_encoder()

        model_macd = ModelMACD()
        enc_macd = model_macd.get_encoder()

        model_mi = ModelMI()
        enc_mi = model_mi.get_encoder()

        model_oscv = ModelOSCV()
        enc_oscv = model_oscv.get_encoder()

        model_psy = ModelPSY()
        enc_psy = model_psy.get_encoder()

        model_roc = ModelROC()
        enc_roc = model_roc.get_encoder()

        model_tr = ModelTR()
        enc_tr = model_tr.get_encoder()

        model_wvad = ModelWVAD()
        enc_wvad = model_wvad.get_encoder()

        model_rsi = ModelRSI()
        enc_rsi = model_rsi.get_encoder()

        model_mdi = ModelMDI()
        enc_mdi = model_mdi.get_encoder()

        input_models = [
            enc_amp, enc_asi, enc_bias, enc_boll, enc_cci,
            enc_change, enc_count, enc_ema, enc_emv, enc_kdj,
            enc_macd, enc_mi, enc_oscv, enc_psy, enc_roc, enc_tr, enc_wvad,
            enc_rsi, enc_mdi
        ]

        self._model = Sequential([
            Merge(input_models, mode='concat',
                  concat_axis=-1,
                  name="dnn_merge_1"),

            Dense(4096,
                  name="dnn_dense_2_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Dense(512,
                  name="dnn_dense_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Dense(256,
                  name="dnn_dense_3"),
            Dropout(0.4),
            BatchNormalization(),
            Activation('relu'),
            Dense(128,
                  name="dnn_dense_3_2"),
            Dropout(0.4),
            BatchNormalization(),
            Activation('relu'),
            Dense(32,
                  name="dnn_dense_4"),
            Dropout(0.2),
            BatchNormalization(),
            Activation('relu'),

            Dense(self.result_categores),
            Activation('softmax')
        ])

        print("Network output layout")
        for layer in self._model.layers:
            print(layer.name, layer.output_shape)
        print("\n\n")
        # exit(0)

        try:
            # self._model = load_model(self._name)
            self._model.load_weights(self._weight_file, by_name=True)
        except Exception:
            pass

        import keras.backend as K
        def max_absolute_error(y_true, y_pred):
            return K.max(K.abs(y_pred - y_true))

        def max_squared_error(y_true, y_pred):
            return K.max(K.abs(y_pred - y_true)) - K.mean(K.abs((y_pred - y_true)), axis=-1)

        def loss(y_true, y_pred):
            return K.max(K.abs(y_pred - y_true)) - K.mean(K.abs(y_pred - y_true))

        def active(y_true, y_pred):
            return K.max(y_pred) - K.min(y_pred)

        def max_pred(y_true, y_pred):
            return K.max(y_pred)

        def min_pred(y_true, y_pred):
            return K.min(y_pred)

        rmsprop = RMSprop(lr=1e-9, rho=0.7, epsilon=1e-4, decay=1e-10)
        sgd = SGD(lr=1e-6, decay=1e-7, momentum=0.5, nesterov=True)

        self._model.compile(optimizer='adadelta',  # adadelta
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        return

    def _transform_inputs(self, input):
        # input = input.reshape(input.shape[0], 1, input.shape[1], input.shape[2])

        ema_in = input[:, :, [12, 13, 14, 15]]
        boll_in = input[:, :, [16, 17, 18]]
        cci_in = input[:, :, [27, 28, 29]]
        rsi_in = input[:, :, [30, 31, 32]]
        kdj_in = input[:, :, [33, 34, 35]]
        bias_in = input[:, :, [36, 37, 38]]
        roc_in = input[:, :, [39, 40]]
        change_in = input[:, :, [41]]
        count_in = input[:, :, [20]]
        amp_in = input[:, :, [42, 43, 44]]
        mi_in = input[:, :, [48, 49, 50, 51]]
        tr_in = input[:, :, [19]]
        oscv_in = input[:, :, [52]]
        mdi_in = input[:, :, [57, 58, 59, 60]]
        asi_in = input[:, :, [61, 62, 63, 64]]
        macd_in = input[:, :, [65, 66, 67]]
        psy_in = input[:, :, [68, 69]]
        emv_in = input[:, :, [70, 71]]
        wvad_in = input[:, :, [72, 73]]

        input = [
            amp_in, asi_in, bias_in, boll_in, cci_in,
            change_in, count_in, ema_in, emv_in, kdj_in,
            macd_in, mi_in, oscv_in, psy_in, roc_in, tr_in, wvad_in,
            rsi_in, mdi_in
        ]

        return input

    def data_features(self):
        features = t5m.features()
        for i in range(len(features)):
            print("{} - {}".format(i, features[i]))
        return

    def scale_result(self, dataset):
        # dataset[:, 0] = (dataset[:, 0] - self.result_min) / (self.result_max - self.result_min)
        # dataset[:, 0] *= 10
        new_dataset = np.zeros((dataset.shape[0], self.result_categores)).astype(np.int8)
        for i in range(dataset.shape[0]):
            value = dataset[i, 0]
            classes = np.linspace(self.result_min, self.result_max, self.result_categores + 1)
            for cls_i in range(len(classes) - 1):
                low = classes[cls_i]
                high = classes[cls_i + 1]
                # print(low, high, value)
                if low <= value < high:
                    v = cls_i
            value_c = np_utils.to_categorical(v, self.result_categores)
            new_dataset[i] = value_c

        print("{0} Samples distributions: ".format(new_dataset.shape[0]))
        print(np.sum(new_dataset, axis=0))
        return new_dataset

    def rev_scale_result(self, dataset):

        # dataset[:, 0] *= 0.1
        # dataset[:, 0] = dataset[:, 0] * (self.result_max - self.result_min) + self.result_min
        return dataset

    def train(self, training_set, validation_set, test_set):
        batch_size = 32

        X_train, X_validation, X_test = training_set[0], validation_set[0], test_set[0]
        y_train, y_validation, y_test = training_set[1], validation_set[1], test_set[1]

        X_train = self._transform_inputs(X_train)
        X_validation = self._transform_inputs(X_validation)
        X_test = self._transform_inputs(X_test)

        y_train = self.scale_result(y_train)
        y_validation = self.scale_result(y_validation)
        y_test = self.scale_result(y_test)

        pd.set_option('display.max_rows', len(y_validation))

        print(y_train.shape)
        print(np.random.choice(y_train[:, 0], size=30, replace=False))
        print("-" * 30)
        print(np.random.choice(y_validation[:, 0], size=30, replace=False))
        print("-" * 30)
        print(np.random.choice(y_test[:, 0], size=30, replace=False))
        print("-" * 30)

        callbacks = [
            EarlyStopping(monitor='val_loss', min_delta=0.05, patience=3, verbose=1, mode='min')
        ]

        print('Training ------------')
        loss, accuracy, sh = 100, 0, 0
        # Another way to train the model
        retry = 0
        # while accuracy <= 0.98:
        while loss >= 0.0005:  # 0.02:
            self._model.fit(X_train, y_train,
                            nb_epoch=2,
                            batch_size=batch_size,
                            callbacks=callbacks,
                            verbose=1,
                            validation_data=(X_validation, y_validation),
                            shuffle=True
                            )

            print('\nTesting ------------')
            loss, acc = self._model.evaluate(X_test, y_test, batch_size=batch_size)

            print('\n\nloss: ', loss)
            print('test acc: ', acc)
            # print('test mean_absolute_error: ', mae)
            # print('test act: {}\tmax: {}\tmin: {} '.format(act, max, min))

            # self._model.save(self._model_file)
            self._model.save_weights(self._weight_file)
            print("model saved\n\n")
            retry += 1
            if retry > 20:
                break
        return loss, accuracy

    def predict(self, data_set):
        X = self._transform_inputs(data_set)
        result = self._model.predict(X, verbose=0)
        # result = self.rev_scale_result(result)
        return result
