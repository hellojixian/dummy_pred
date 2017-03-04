from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Convolution2D, BatchNormalization, Merge
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.regularizers import l1l2, activity_l1l2
import pandas as pd
import numpy as np
import Common.config as config
import os

from DataTransform.Transform5M import Transform5M as t5m


class Model_CNN_NDC:
    def __init__(self):
        name = "Model_CNN_NDC"
        self._model_file = os.path.join(config.MODEL_DIR, name + '_model.h5')
        self._weight_file = os.path.join(config.MODEL_DIR, name + '_weight.h5')
        data_len = 48

        price_vec_dim = 4
        price_change_dim = 4
        ma_dim = 4
        ema_dim = 4
        boll_dim = 3
        vol_dim = 8
        cci_dim = 3
        rsi_dim = 3
        kdj_dim = 3
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

        price_vec_model = Sequential([
            Convolution2D(88, 5, price_vec_dim, border_mode='valid',
                          input_shape=(1, data_len, price_vec_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="price_vec_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="price_vec_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="price_vec_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        price_change_model = Sequential([
            Convolution2D(88, 5, price_change_dim, border_mode='valid',
                          input_shape=(1, data_len, price_change_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="price_change_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="price_change_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="price_change_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        ma_model = Sequential([
            Convolution2D(88, 5, ma_dim, border_mode='valid',
                          input_shape=(1, data_len, ma_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="ma_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="ma_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="ma_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        ema_model = Sequential([
            Convolution2D(88, 5, ema_dim, border_mode='valid',
                          input_shape=(1, data_len, ema_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="ema_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="ema_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="ema_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        boll_model = Sequential([
            Convolution2D(88, 5, boll_dim, border_mode='valid',
                          input_shape=(1, data_len, boll_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="boll_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="boll_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="boll_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        vol_model = Sequential([
            Convolution2D(146, 5, vol_dim, border_mode='valid',
                          input_shape=(1, data_len, vol_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="vol_conv_0"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(64, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="vol_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="vol_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="vol_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        cci_model = Sequential([
            Convolution2D(88, 5, cci_dim, border_mode='valid',
                          input_shape=(1, data_len, cci_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="cci_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="cci_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="cci_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        rsi_model = Sequential([
            Convolution2D(88, 5, rsi_dim, border_mode='valid',
                          input_shape=(1, data_len, rsi_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="rsi_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="rsi_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="rsi_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        kdj_model = Sequential([
            Convolution2D(88, 5, kdj_dim, border_mode='valid',
                          input_shape=(1, data_len, kdj_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="kdj_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="kdj_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="kdj_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        bias_model = Sequential([
            Convolution2D(88, 5, bias_dim, border_mode='valid',
                          input_shape=(1, data_len, bias_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="bias_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="bias_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="bias_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        roc_model = Sequential([
            Convolution2D(88, 5, roc_dim, border_mode='valid',
                          input_shape=(1, data_len, roc_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="roc_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="roc_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="roc_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        change_model = Sequential([
            Convolution2D(88, 5, change_dim, border_mode='valid',
                          input_shape=(1, data_len, change_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="change_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="change_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="change_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        amp_model = Sequential([
            Convolution2D(88, 5, amp_dim, border_mode='valid',
                          input_shape=(1, data_len, amp_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="amp_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="amp_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="amp_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        wr_model = Sequential([
            Convolution2D(88, 5, wr_dim, border_mode='valid',
                          input_shape=(1, data_len, wr_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="wr_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="wr_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="wr_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        mi_model = Sequential([
            Convolution2D(88, 5, mi_dim, border_mode='valid',
                          input_shape=(1, data_len, mi_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="mi_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="mi_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="mi_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        oscv_model = Sequential([
            Convolution2D(88, 5, oscv_dim, border_mode='valid',
                          input_shape=(1, data_len, oscv_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="oscv_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="oscv_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="oscv_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        dma_model = Sequential([
            Convolution2D(88, 5, dma_dim, border_mode='valid',
                          input_shape=(1, data_len, dma_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="dma_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="dma_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="dma_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        abr_model = Sequential([
            Convolution2D(88, 5, abr_dim, border_mode='valid',
                          input_shape=(1, data_len, abr_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="abr_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="abr_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="abr_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        mdi_model = Sequential([
            Convolution2D(88, 5, mdi_dim, border_mode='valid',
                          input_shape=(1, data_len, mdi_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="mdi_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="mdi_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="mdi_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        asi_model = Sequential([
            Convolution2D(88, 5, asi_dim, border_mode='valid',
                          input_shape=(1, data_len, asi_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="asi_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="asi_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="asi_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        macd_model = Sequential([
            Convolution2D(88, 5, macd_dim, border_mode='valid',
                          input_shape=(1, data_len, macd_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="macd_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="macd_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="macd_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        psy_model = Sequential([
            Convolution2D(88, 5, psy_dim, border_mode='valid',
                          input_shape=(1, data_len, psy_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="psy_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="psy_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="psy_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        emv_model = Sequential([
            Convolution2D(88, 5, emv_dim, border_mode='valid',
                          input_shape=(1, data_len, emv_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="emv_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="emv_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="emv_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])
        wvad_model = Sequential([
            Convolution2D(88, 5, wvad_dim, border_mode='valid',
                          input_shape=(1, data_len, wvad_dim),
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="wvad_conv_1"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(32, 8, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="wvad_conv_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Convolution2D(16, 1, 1, border_mode='valid',
                          dim_ordering='th',
                          W_regularizer=l1l2(l1=0.05, l2=0.05),
                          b_regularizer=l1l2(l1=0.05, l2=0.05),
                          activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                          name="wvad_conv_3"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.4),
            Flatten(),
        ])

        # input_models = [
        #     price_vec_model, price_change_model,  ema_model,
        #     boll_model, vol_model, cci_model, rsi_model, kdj_model, bias_model, roc_model,
        #     change_model, amp_model, wr_model, mi_model, oscv_model, dma_model, abr_model,
        #     mdi_model, asi_model, macd_model, psy_model, emv_model, wvad_model
        # ]
        input_models = [
            price_vec_model, price_change_model, ema_model,
            boll_model, vol_model, cci_model, kdj_model, bias_model,
            change_model, amp_model,macd_model
        ]
        self._model = Sequential([
            Merge(input_models, mode='concat',
                  concat_axis=-1,
                  name="dnn_merge_1"),
            Dense(2048,
                  W_regularizer=l1l2(l1=0.05, l2=0.05),
                  b_regularizer=l1l2(l1=0.05, l2=0.05),
                  activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                  name="dnn_dense_0_2"),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.5),
            # Dense(512, init='normal',
            #       W_regularizer=l1l2(l1=0.05, l2=0.05),
            #       b_regularizer=l1l2(l1=0.05, l2=0.05),
            #       activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
            #       name="dnn_dense_2"),
            # BatchNormalization(),
            # Activation('relu'),
            # Dropout(0.4),
            Dense(256, init='normal',
                  W_regularizer=l1l2(l1=0.05, l2=0.05),
                  b_regularizer=l1l2(l1=0.05, l2=0.05),
                  activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                  name="dnn_dense_3"),
            Dropout(0.4),
            BatchNormalization(),
            Activation('relu'),
            Dense(32, init='normal',
                  W_regularizer=l1l2(l1=0.05, l2=0.05),
                  b_regularizer=l1l2(l1=0.05, l2=0.05),
                  activity_regularizer=activity_l1l2(l1=0.05, l2=0.05),
                  name="dnn_dense_4"),
            BatchNormalization(),
            Dense(1)
        ])

        # print("Network output layout")
        # for layer in self._model.layers:
        #     print(layer.output_shape)
        # print("\n\n")
        # exit(0)

        try:
            # self._model = load_model(self._name)
            self._model.load_weights(self._weight_file, by_name=True)
        except Exception:
            pass

        rmsprop = RMSprop(lr=1e-9, rho=0.7, epsilon=1e-4, decay=1e-10)
        sgd = SGD(lr=1e-6, decay=1e-7, momentum=0.5, nesterov=True)

        self._model.compile(optimizer='adadelta',  # adadelta
                            loss='mse',
                            metrics=['mae', 'squared_hinge', 'hinge'])
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

        # input = [
        #     price_vec_in, price_change_in, ema_in,
        #     boll_in, vol_in, cci_in, rsi_in, kdj_in, bias_in, roc_in,
        #     change_in, amp_in, wr_in, mi_in, oscv_in, dma_in, abr_in,
        #     mdi_in, asi_in, macd_in, psy_in, emv_in, wvad_in
        # ]
        input = [
            price_vec_in, price_change_in, ema_in,
            boll_in, vol_in, cci_in, kdj_in, bias_in,
            change_in, amp_in, macd_in
        ]
        return input

    def train(self, training_set, validation_set, test_set):
        batch_size = 32

        X_train, X_validation, X_test = training_set[0], validation_set[0], test_set[0]
        y_train, y_validation, y_test = training_set[1], validation_set[1], test_set[1]

        X_train = self._transform_inputs(X_train)
        X_validation = self._transform_inputs(X_validation)
        X_test = self._transform_inputs(X_test)

        callbacks = [
            EarlyStopping(monitor='val_loss', min_delta=0.05, patience=3, verbose=1, mode='min')
        ]

        print('Training ------------')
        loss, accuracy, sh, h = 100, 0, 0, 0
        # Another way to train the model
        retry = 0
        # while accuracy <= 0.98:
        while loss >= 0.5:  # 0.02:
            self._model.fit(X_train, y_train,
                            nb_epoch=2,
                            batch_size=batch_size,
                            callbacks=callbacks,
                            verbose=1,
                            validation_data=(X_validation, y_validation),
                            shuffle=True
                            )

            print('\nTesting ------------')
            loss, accuracy, sh, h = self._model.evaluate(X_test, y_test, batch_size=batch_size)
            print('\ntest mean_squared_error: ', loss)
            print('test mean_absolute_error: ', accuracy)
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
        return result
