# http://baike.baidu.com/item/MACD指标
# 输出  DIF, DEA, MACD

import pandas as pd
import numpy as np
from Common.MathFunctions import EMA


def calculate(df):
    n_short = 12
    n_long = 26
    dif_list, dea_list, macd_list = [], [], []
    for i in range(0, df.shape[0]):
        close_list = df.iloc[i - n_long: i, df.columns.get_loc('close')]
        if len(close_list) > 0:
            ema_short = EMA(close_list, n_short)
            ema_long = EMA(close_list, n_long)
        else:
            ema_short = 0
            ema_long = 0
        dif = ema_short - ema_long
        if i == 0:
            dea_last = 0
        else:
            dea_last = dea_list[i - 1]
        dea = dea_last * 0.8 + dif * 0.2
        macd = (dif - dea) * 2
        dif_list.append(dif)
        dea_list.append(dea)
        macd_list.append(macd)

    df['macd_dif'] = pd.Series(dif_list, index=df.index)
    df['macd_dea'] = pd.Series(dea_list, index=df.index)
    df['macd_bar'] = pd.Series(macd_list, index=df.index)
    return df
