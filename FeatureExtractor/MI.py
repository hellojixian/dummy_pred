# http://www.taindicators.com/2010/03/momentum-indicator.html
# Momentum Indicator 動量指標

import pandas as pd
import numpy as np
import math


def calculate(df):
    # 動量指標
    # Momentum = 即日收巿價 － n天前收巿價
    # 返回结果可能有负数，差距不会很大，通常不用缩放，
    # 但是如果在分钟线上，估计还是要放大10倍 看的更清楚一点

    mi_steps = [5, 10, 20, 30]
    mi_matrix = np.zeros((len(mi_steps), df.shape[0]), dtype=np.float32)

    for i in range(0, df.shape[0]):
        index = df.index[i]
        close = df.loc[index, 'close']

        for j in range(0, len(mi_steps)):
            step = mi_steps[j]
            close_n = np.max(df.iloc[i - step, df.columns.get_loc('close')])

            mi_n = close - close_n
            mi_matrix[j, i] = mi_n

    for i in range(0, len(mi_steps)):
        step = mi_steps[i]
        df['mi_' + str(step)] = pd.Series(mi_matrix[i], index=df.index)
    return df
