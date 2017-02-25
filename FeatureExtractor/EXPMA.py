# http://baike.baidu.com/item/指数平均数/3334858?fromtitle=EXPMA指标&fromid=8517919
# 输出 短期和长期 5, 15, 25, 40


import pandas as pd
import numpy as np
import math


def calculate(df):
    # ema_n = ema_last + 2/(n+1) * (close-ema_last)

    ema_steps = [5, 15, 25, 40]
    ema_matrix = np.zeros((len(ema_steps), df.shape[0]), dtype=np.float32)

    idx_close = df.columns.get_loc('close')

    for i in range(0, df.shape[0]):
        close = df.iloc[i, idx_close]
        for j in range(0, len(ema_steps)):
            step = ema_steps[j]
            if i == 0:
                ema_last = close
            else:
                ema_last = ema_matrix[j, i-1]
            ema = ema_last + 2 / (step + 1) * (close - ema_last)
            ema_matrix[j, i] = ema

    for i in range(0, len(ema_steps)):
        step = ema_steps[i]
        df['ema_' + str(step)] = pd.Series(ema_matrix[i], index=df.index)

    return df

