import pandas as pd
import numpy as np


def calculate(df):
    # n日RSV=（Cn－Ln）/（Hn－Ln）×100

    k_list, d_list, j_list = [], [], []
    k, d, j = 50, 50, 50
    n = 9

    for i in range(0, df.shape[0]):
        high = np.max(df.iloc[i - n:i, df.columns.get_loc('high')])
        low = np.min(df.iloc[i - n:i, df.columns.get_loc('low')])
        close = df.iloc[i, df.columns.get_loc('close')]

        if i > n:
            if high != low:
                rsv = (close - low) / (high - low) * 100
                k = 2 / 3 * k_list[i - 1] + 1 / 3 * rsv
                d = 2 / 3 * d_list[i - 1] + 1 / 3 * k
                j = 3 * k - 2 * d
            else:
                k, d, j = 50, 50, 50
        k_list.append(k)
        d_list.append(d)
        j_list.append(j)

    df['k9'] = pd.Series(k_list, index=df.index)
    df['d9'] = pd.Series(d_list, index=df.index)
    df['j9'] = pd.Series(j_list, index=df.index)

    return df
