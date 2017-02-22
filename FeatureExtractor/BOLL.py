import pandas as pd
import numpy as np
import math


def calculate(df):
    # （1）计算MA
    # MA = N日内的收盘价之和÷N
    # （2）计算标准差MD
    # MD = 平方根（N - 1）日的（C－MA）的两次方之和除以N
    # （3）计算MB、UP、DN线
    # MB =（N－1）日的MA
    # UP = MB + k×MD
    # DN = MB－k×MD
    # （K为参数，可根据股票的特性来做相应的调整，一般默认为2）

    n = 20
    k = 2
    mb_list, up_list, dn_list = [],[],[]
    for i in range(0, df.shape[0]):
        ma = np.mean(df.iloc[i - n:i, df.columns.get_loc('close')])
        c = np.sum((df.iloc[i - n - 1:i, df.columns.get_loc('close')] - ma) ** 2) / n
        md = math.sqrt(c)

        mb = np.mean(df.iloc[i - n - 1:i, df.columns.get_loc('close')])
        up = mb + k * md
        dn = mb - k * md

        mb_list.append(mb)
        up_list.append(up)
        dn_list.append(dn)

    df['boll_up'] = pd.Series(up_list, index=df.index)
    df['boll_md'] = pd.Series(mb_list, index=df.index)
    df['boll_dn'] = pd.Series(dn_list, index=df.index)
    return df
