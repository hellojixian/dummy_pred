# https://zhidao.baidu.com/question/264616493.html


import pandas as pd
import numpy as np
import math


def calculate(df):
    # DIF：ma（c，n1）- ma（c，n2）
    # AMA：ma（dif，m）

    n1 = 10
    n2 = 50
    m = 10

    dif_list = []
    ama_list = []

    idx_close = df.columns.get_loc('close')
    for i in range(0, df.shape[0]):
        close = df.iloc[i, idx_close]
        ma_n1 = float(np.mean(df.iloc[i - n1:i, idx_close]))
        ma_n2 = float(np.mean(df.iloc[i - n2:i, idx_close]))
        dif = ma_n1 - ma_n2
        dif_list.append(dif)
    df['dma_dif'] = pd.Series(dif_list, index=df.index)

    idx_dif = df.columns.get_loc('dma_dif')
    for i in range(0, df.shape[0]):
        ama = float(np.mean(df.iloc[i - m:i, idx_dif]))
        ama_list.append(dif)
    df['dma_ama'] = pd.Series(ama_list, index=df.index)
    return df
