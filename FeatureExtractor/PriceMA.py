import pandas as pd
import numpy as np

def calculate(df):
    # 计算收盘价的N日均线
    # calculate ma5, ma10, ma20, ma30
    ma_steps = [5, 15, 25, 40]
    ma_matrix = np.zeros((len(ma_steps), df.shape[0]), dtype=np.float32)
    idx_close = df.columns.get_loc('close')
    for i in range(0, df.shape[0]):
        for j in range(0, len(ma_steps)):
            step = ma_steps[j]
            ma_matrix[j, i] = float(np.mean(df.iloc[i - step:i,
                                            idx_close]))

    for i in range(0, len(ma_steps)):
        step = ma_steps[i]
        df['ma' + str(step)] = pd.Series(ma_matrix[i], index=df.index)

    return df
