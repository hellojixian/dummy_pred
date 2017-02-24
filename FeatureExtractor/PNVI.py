# http://www.taindicators.com/2010/03/nvi-negative-volume-index.html


import pandas as pd
import numpy as np
import math


def calculate(df):
    # NVI - Negative Volume Index 負量指標
    # if vol < vol_last:
    #    nvi = nvi_last + (close - close_last) / close_last * nvi_last
    # else:
    #    nvi = nvi_last

    nvi_series = []
    pvi_series = []

    idx_vol = df.columns.get_loc('vol')
    idx_close = df.columns.get_loc('close')
    for i in range(0, df.shape[0]):
        close_last = df.iloc[i - 1, idx_close]
        vol_last = df.iloc[i - 1, idx_vol]
        close = df.iloc[i, idx_close]
        vol = df.iloc[i, idx_vol]
        nvi_last = nvi_series[i-1]
        pvi_last = pvi_series[i - 1]
        nvi = 0
        pvi = 0
        if vol < vol_last:
            nvi = nvi_last + (close - close_last) / close_last * nvi_last
            pvi = pvi_last
        else:
            nvi = nvi_last
            pvi = pvi_last + (close - close_last) / close_last * pvi_last

        nvi_series.append(nvi)
        pvi_series.append(pvi)

    df['nvi'] = pd.Series(nvi_series, index=df.index)
    df['pvi'] = pd.Series(pvi_series, index=df.index)
    return df
