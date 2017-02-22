import pandas as pd
import numpy as np


def calculate(df):
    # N日RS=[A÷B]×100%
    # A——N日内收盘涨幅之和
    # B——N日内收盘跌幅之和(取正值)
    # RSI_N=100-100/(1+RS)

    rsi_steps = [6, 12, 24]
    rsi_matrix = np.zeros((len(rsi_steps), df.shape[0]), dtype=np.float32)

    for i in range(0, df.shape[0]):
        for j in range(0, len(rsi_steps)):
            step = rsi_steps[j]
            change = df.iloc[i - step:i, df.columns.get_loc('change')].to_frame(name='change')
            change_up = float(np.sum(change[change['change'] > 0]))
            change_down = float(np.abs(np.sum(change[change['change'] < 0])))
            if (change_down > 0):
                rs = change_up / change_down
                rsi_n = 100 - 100 / (1 + rs)
                rsi_matrix[j, i] = rsi_n

    for i in range(0, len(rsi_steps)):
        step = rsi_steps[i]
        df['rsi_' + str(step)] = pd.Series(rsi_matrix[i], index=df.index)
    return df
