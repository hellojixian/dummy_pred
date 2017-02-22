import pandas as pd
import numpy as np


def calculate(df):
    # N日BIAS=（当日收盘价—N日移动平均价）÷N日移动平均价×100

    bias_steps = [5, 10, 30]
    bias_matrix = np.zeros((len(bias_steps), df.shape[0]), dtype=np.float32)

    for i in range(0, df.shape[0]):
        close = df.iloc[i, df.columns.get_loc('close')]
        for j in range(0, len(bias_steps)):
            step = bias_steps[j]

            close_mean_n = np.mean(df.iloc[i - step:i, df.columns.get_loc('close')])
            bias_n = (close - close_mean_n) / close_mean_n * 100
            bias_matrix[j, i] = bias_n

    for i in range(0, len(bias_steps)):
        step = bias_steps[i]
        df['bias_' + str(step)] = pd.Series(bias_matrix[i], index=df.index)
    return df
