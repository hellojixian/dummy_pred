# http://baike.baidu.com/item/EMV指标
# 输出 EMV 和 MAEMV


import pandas as pd
import numpy as np
import math


def calculate(df):
    # A = (high-low)/2
    # B = (high_last - low_last)/2
    # C = high - low
    # em = (a-b)*c/amount
    # emv = ma(em,n)
    # maemv = ma(emv,m)
    # n = 14
    # m = 9

    n = 14
    m = 9

    em_list = []
    emv_list = []
    maemv_list = []

    idx_high = df.columns.get_loc('high')
    idx_low = df.columns.get_loc('low')
    idx_vol = df.columns.get_loc('vol')

    for i in range(0, df.shape[0]):
        high = df.iloc[i, idx_high]
        high_last = df.iloc[i - 1, idx_high]
        low = df.iloc[i, idx_low]
        low_last = df.iloc[i - 1, idx_low]
        amount = df.iloc[i, idx_vol]

        a = (high - low) / 2
        b = (high_last - low_last) / 2
        c = high - low
        if amount != 0:
            em = (a - b) * c / amount
        else:
            em = 0
        em *= 1000000
        em_list.append(em)
    df['emv_em'] = pd.Series(em_list, index=df.index)


    idx_em = df.columns.get_loc('emv_em')
    for i in range(0, df.shape[0]):
        emv = np.sum(df.iloc[i - n:i, idx_em])
        emv_list.append(emv)
    df['emv_emv'] = pd.Series(emv_list, index=df.index)

    idx_emv = df.columns.get_loc('emv_emv')
    for i in range(0, df.shape[0]):
        maemv = float(np.mean(df.iloc[i - m:i, idx_emv]))
        maemv_list.append(maemv)
    df['emv_maemv'] = pd.Series(maemv_list, index=df.index)

    df = df.drop(labels="emv_em", axis=1)
    return df
