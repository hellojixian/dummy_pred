import pandas as pd
import numpy as np

def calculate(df):
    # 1、 AX=今天的收盘价—12天前的收盘价
    # 2、 BX=12天前的收盘价
    # 3、 ROC=AX/BX

    roc_steps = [12,25]
    roc_matrix = np.zeros((len(roc_steps), df.shape[0]), dtype=np.float32)

    for i in range(0, df.shape[0]):
        index = df.index[i]
        close = df.loc[index, 'close']
        for j in range(0, len(roc_steps)):
            step = roc_steps[j]
            bx = df.iloc[i - step, df.columns.get_loc('close')]
            ax = close - bx
            roc_n = ax/bx
            roc_matrix[j, i] = roc_n

    for i in range(0, len(roc_steps)):
        step = roc_steps[i]
        df['roc_' + str(step)] = pd.Series(roc_matrix[i], index=df.index)
    return df