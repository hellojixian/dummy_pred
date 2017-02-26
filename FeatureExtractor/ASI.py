# http://baike.baidu.com/item/ASI指标
# 输出 ASI 曲线, 设置四个观察数值

import pandas as pd
import numpy as np


def calculate(df):
    si_list = []

    asi_steps = [5, 15, 25, 40]
    asi_matrix = np.zeros((len(asi_steps), df.shape[0]), dtype=np.float32)

    for i in range(0, df.shape[0]):
        open = df.iloc[i, df.columns.get_loc('open')]
        high = df.iloc[i, df.columns.get_loc('high')]
        low = df.iloc[i, df.columns.get_loc('low')]
        close = df.iloc[i, df.columns.get_loc('close')]

        close_last = df.iloc[i - 1:i, df.columns.get_loc('close')]
        open_last = df.iloc[i - 1:i, df.columns.get_loc('open')]
        low_last = df.iloc[i - 1:i, df.columns.get_loc('low')]

        if i == 0:
            si_list.append(0)
            continue

        a = float(np.abs(high - close_last))
        b = float(np.abs(low - close_last))
        c = float(np.abs(high - low_last))
        d = float(np.abs(close_last - open_last))

        max_v = np.max([a, b, c])
        if max_v == a:
            r = a + 0.5 * b + 0.25 * d
        elif max_v == b:
            r = b + 0.5 * a + 0.25 * d
        elif max_v == c:
            r = c + 0.25 * d
        e = close - close_last
        f = close - open
        g = close_last - open_last
        x = e + 0.5 * f + g
        k = np.max([a, b])
        l = 3
        si = 50 * x / r * k / l
        si = float(si)
        si_list.append(si)

    for i in range(0, df.shape[0]):
        for j in range(0, len(asi_steps)):
            step = asi_steps[j]
            asi_n = np.sum(si_list[i - step:i])
            asi_matrix[j, i] = asi_n

    for i in range(0, len(asi_steps)):
        step = asi_steps[i]
        df['asi_' + str(step)] = pd.Series(asi_matrix[i], index=df.index)
    return df
