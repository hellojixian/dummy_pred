# http://www.taindicators.com/2010/03/oscv-volume-oscillator.html
# OSCV - Volume Oscillator 成交量擺動指標



import pandas as pd
import numpy as np
import math


def calculate(df):
    # OSCV - Volume Oscillator 成交量擺動指標
    # oscp = ( vol_ma_min - vol_ma_max ) / vol_ma_min * 100

    oscv_min = 10
    oscv_max = 30

    oscv_list = []
    idx_vol = df.columns.get_loc('vol')
    for i in range(0, df.shape[0]):
        vol_ma_min = float(np.mean(df.iloc[i - oscv_min:i, idx_vol]))
        vol_ma_max = float(np.mean(df.iloc[i - oscv_max:i, idx_vol]))
        if vol_ma_min != 0:
            oscv = (vol_ma_min - vol_ma_max) / vol_ma_min * 100
        else:
            oscv = 0
        oscv_list.append(oscv)

    df['oscv'] = pd.Series(oscv_list, index=df.index)
    return df
