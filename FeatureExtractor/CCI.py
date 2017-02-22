import pandas as pd
import numpy as np
import math


def calculate(df):
    # TYP: = (HIGH + LOW + CLOSE) / 3;
    # MA = MA(TYP, N))
    # MD = AVEDEV(TYP, N)
    # CCI: (TYP - MA / (0.015 * MD);

    magic_rate = 0.015
    cci_steps = [5, 15, 30]
    cci_matrix = np.zeros((len(cci_steps), df.shape[0]), dtype=np.float32)

    for i in range(0, df.shape[0]):
        index = df.index[i]
        high, low, close = df.loc[index, ['high', 'low', 'close']]
        # print(index, high, low, close)
        tp = (high + low + close) / 3
        for j in range(0, len(cci_steps)):
            step = cci_steps[j]
            ma_n = np.sum(np.mean(df.iloc[i - step:i,
                                  [df.columns.get_loc('high'), df.columns.get_loc('low'),
                                   df.columns.get_loc('close')]])) / 3
            tp_n = np.sum(df.iloc[i - step:i,
                          [df.columns.get_loc('high'), df.columns.get_loc('low'),
                           df.columns.get_loc('close')]], axis=1) / 3
            md_n = np.mean(np.abs(tp_n - ma_n))

            p = magic_rate * md_n
            # print(p)
            if p > 0:
                cci_n = (tp - ma_n) / p
                cci_matrix[j, i] = cci_n

    for i in range(0, len(cci_steps)):
        step = cci_steps[i]
        df['cci_' + str(step)] = pd.Series(cci_matrix[i], index=df.index)
    return df
