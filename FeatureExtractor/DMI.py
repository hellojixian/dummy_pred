# http://baike.baidu.com/item/DMI指标/3423254?sefr=cr
# http://www.ezchart.com.tw/inds.php?IND=DMI
# 最后输出的是 PDI MDI ADX, ADXR 四组数据

import pandas as pd
import numpy as np


def calculate(df):
    n = 7
    m = 15
    m2 = 21
    pdi_list, mdi_list, adx_list, adxr_list = [], [], [], []
    pdm_list, mdm_list, tr_list, dx_list = [], [], [], []

    for i in range(0, df.shape[0]):
        high = df.iloc[i, df.columns.get_loc('high')]
        high_last = df.iloc[i - 1, df.columns.get_loc('high')]
        low = df.iloc[i, df.columns.get_loc('low')]
        low_last = df.iloc[i - 1, df.columns.get_loc('low')]
        close = df.iloc[i, df.columns.get_loc('close')]
        close_last = df.iloc[i - 1, df.columns.get_loc('close')]

        pdm = high - high_last
        if pdm <= 0:
            pdm = 0
        mdm = low_last - low
        if mdm <= 0:
            mdm = 0

        t = []
        t.append(np.abs(high - close_last))
        t.append(np.abs(close - low))
        t.append(np.abs(low - close_last))
        tr = np.max(t)

        pdm_list.append(pdm)
        mdm_list.append(mdm)
        tr_list.append(tr)

    for i in range(0, df.shape[0]):
        tr_n = np.sum(tr_list[i - n:i])
        pdm_n = np.sum(pdm_list[i - n:i])
        mdm_n = np.sum(mdm_list[i - n:i])
        if tr_n != 0:
            pdi_n = pdm_n / tr_n
            mdi_n = mdm_n / tr_n
        else:
            pdi_n, mdi_n, tr_n = 0, 0, 0
        di_sum = pdi_n + mdm_n
        di_diff = pdi_n - mdm_n
        dx = np.abs(di_diff - di_sum) * 50

        pdi_list.append(pdi_n)
        mdi_list.append(mdi_n)
        dx_list.append(dx)

    for i in range(0, df.shape[0]):
        dx = dx_list[i]
        adx = (np.mean(dx_list[i - m - 1:i - 1]) + dx) / m
        adx_list.append(adx)

    for i in range(0, df.shape[0]):
        adx = adx_list[i]
        adxr = (adx_list[i - m2] + adx) / 2
        adxr_list.append(adxr)

    df['pdi'] = pd.Series(pdi_list, index=df.index)
    df['mdi'] = pd.Series(mdi_list, index=df.index)
    df['adx'] = pd.Series(adx_list, index=df.index)
    df['adxr'] = pd.Series(adxr_list, index=df.index)
    return df
