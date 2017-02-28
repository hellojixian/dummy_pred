# http://baike.baidu.com/item/WVAD%E6%8C%87%E6%A0%87?sefr=cr
# 画多条线 分别是 6, 24 和他们的均线


import pandas as pd
import numpy as np


def calculate(df):
    # n = 12
    # m = 6
    # A=当天收盘价－当天开盘价
    # B=当天最高价－当天最低价
    # V=当天成交量
    # WVAD=∑n（A÷B×V)

    n = 24
    m = 6
    v_list = []
    wvad_list = []
    wvadma_list = []

    for i in range(0, df.shape[0]):
        close = df.iloc[i, df.columns.get_loc('close')]
        open = df.iloc[i, df.columns.get_loc('open')]
        high = df.iloc[i, df.columns.get_loc('high')]
        low = df.iloc[i, df.columns.get_loc('low')]
        vol = df.iloc[i, df.columns.get_loc('vol')]
        a = close - open
        b = high - low
        if b == 0:
            v = 0
        else:
            v = (a / b) * vol
        v_list.append(v)

    for i in range(0, df.shape[0]):
        wvad = np.sum(v_list[i - n:i])
        wvad_list.append(wvad)

    for i in range(0, df.shape[0]):
        if i > m:
            wvadma = np.mean(wvad_list[i - m:i])
        else:
            wvadma = 0
        wvadma_list.append(wvadma)

    df['wvad'] = pd.Series(wvad_list, index=df.index)
    df['wvad_ma'] = pd.Series(wvadma_list, index=df.index)
    return df
