import pandas as pd
import numpy as np

def calculate(df):
    # calculate ma5, ma10, ma20, ma30
    # 计算成交量的N日均线
    ma_steps = [5, 15, 25, 40]
    ma_matrix = np.zeros((len(ma_steps), df.shape[0]), dtype=np.float32)
    idx_vol = df.columns.get_loc('vol')
    for i in range(0, df.shape[0]):
        for j in range(0, len(ma_steps)):
            step = ma_steps[j]
            ma_matrix[j, i] = float(np.mean(df.iloc[i - step:i,
                                            idx_vol]))

    for i in range(0, len(ma_steps)):
        step = ma_steps[i]
        df['v_ma' + str(step)] = pd.Series(ma_matrix[i], index=df.index)

    return df