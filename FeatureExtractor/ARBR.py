# http://baike.baidu.com/item/arbr指标/2042243
# 输出 AR BR 两条曲线
# 缩放的时候可以判断两个指标距离 100 的距离


import pandas as pd
import numpy as np


def calculate(df):
    # n = 26
    # ar = ma(high - open, n) / ma(open - low ,n )* 100
    # br = ma(high - close_last,n) / ma(close_last - low, n)* 100

    n = 26
    ar_list = []
    br_list = []

    for i in range(0, df.shape[0]):
        ma_open = np.mean(df.iloc[i - n:i, df.columns.get_loc('open')])
        ma_high = np.mean(df.iloc[i - n:i, df.columns.get_loc('high')])
        ma_low = np.mean(df.iloc[i - n:i, df.columns.get_loc('low')])
        ma_close_last = np.mean(df.iloc[i - n - 1:i - 1, df.columns.get_loc('close')])

        ar, br = 0, 0
        if (ma_open - ma_low) != 0:
            ar = (ma_high - ma_open) / (ma_open - ma_low) * 100

        if (ma_close_last - ma_low) != 0:
            br = (ma_high - ma_close_last) / (ma_close_last - ma_low) * 100

        ar_list.append(ar)
        br_list.append(br)

    df['ar'] = pd.Series(ar_list, index=df.index)
    df['br'] = pd.Series(br_list, index=df.index)
    return df
