# http://baike.baidu.com/item/PSY%E6%8C%87%E6%A0%87



import pandas as pd
import numpy as np


def calculate(df):
    # n = 12
    # m = 6
    # 1.PSY = N日内上涨天数 / N * 100
    # len(np.where(a>60)[0])
    # 2.PSYMA = PSY的M日简单移动平均
    # 3.参数N设置为12日，参数M设置为6日

    n = 12
    m = 6
    psy_list = []
    psyma_list = []

    for i in range(0, df.shape[0]):
        psy = len(np.where(df.iloc[i - n:i, df.columns.get_loc('change')] > 0)[0])
        psy = psy /n * 100
        psy_list.append(psy)

    for i in range(0, df.shape[0]):
        psyma = np.mean(psy_list[i - m:i])
        psyma_list.append(psyma)

    df['psy'] = pd.Series(psy_list, index=df.index)
    df['psy_ma'] = pd.Series(psyma_list, index=df.index)
    return df
