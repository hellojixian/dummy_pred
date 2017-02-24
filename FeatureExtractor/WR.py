# http://baike.baidu.com/link?url=04BjJWoNPdR5caUtx1BGF9Baul3qxxSkosENspCf_48YA0sHrml6HV5885ciOf3DgX3c-TZrHrqa6VoiGSL2HvAEbZ-jmHFwOSszn1TzTDi


import pandas as pd
import numpy as np
import math


def calculate(df):
    # 以N日威廉指标为例，
    # WR(N) = 100 * [HIGH(N) - C] / [HIGH(N) - LOW(N)]
    # C：当日收盘价
    # HIGH(N)：N日内的最高价
    # LOW(n)：N日内的最低价

    wr_steps = [5, 10, 20]
    wr_matrix = np.zeros((len(wr_steps), df.shape[0]), dtype=np.float32)

    for i in range(0, df.shape[0]):
        index = df.index[i]
        close = df.loc[index, 'close']

        for j in range(0, len(wr_steps)):
            step = wr_steps[j]
            high_n = np.max(df.iloc[i - step:i, df.columns.get_loc('high')])
            low_n = np.min(df.iloc[i - step:i, df.columns.get_loc('low')])
            p = high_n / low_n
            if p != 0:
                wr_n = 100 * (high_n - close) / ( high_n - low_n )
                wr_matrix[j, i] = wr_n

    for i in range(0, len(wr_steps)):
        step = wr_steps[i]
        df['wr_' + str(step)] = pd.Series(wr_matrix[i], index=df.index)
    return df
