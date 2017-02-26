import pandas as pd
import numpy as np


def calculate(df):
    price_vec_matrix = np.zeros((8, df.shape[0]))
    idx_high = df.columns.get_loc('high')
    idx_low = df.columns.get_loc('low')
    idx_close = df.columns.get_loc('close')
    idx_open = df.columns.get_loc('open')
    for i in range(0, df.shape[0]):
        if i > 0:
            last_open_price = df.iloc[i - 1, idx_open]
            last_close_price = df.iloc[i - 1, idx_close]
            last_high_price = df.iloc[i - 1, idx_high]
            last_low_price = df.iloc[i - 1, idx_low]
        else:
            last_open_price = df.iloc[i, idx_open]
            last_close_price = df.iloc[i, idx_close]
            last_high_price = df.iloc[i, idx_high]
            last_low_price = df.iloc[i, idx_low]
        mean_price = np.mean([df.iloc[i, idx_close],
                              df.iloc[i, idx_open],
                              df.iloc[i, idx_high],
                              df.iloc[i, idx_low]])
        price_vec_matrix[0, i] = (df.iloc[i, idx_open] - mean_price) / mean_price
        price_vec_matrix[1, i] = (df.iloc[i, idx_close] - mean_price) / mean_price
        price_vec_matrix[2, i] = (df.iloc[i, idx_high] - mean_price) / mean_price
        price_vec_matrix[3, i] = (df.iloc[i, idx_low] - mean_price) / mean_price

        price_vec_matrix[4, i] = (df.iloc[i, idx_open] - last_open_price) / last_open_price
        price_vec_matrix[5, i] = (df.iloc[i, idx_close] - last_close_price) / last_close_price
        price_vec_matrix[6, i] = (df.iloc[i, idx_high] - last_high_price) / last_high_price
        price_vec_matrix[7, i] = (df.iloc[i, idx_low] - last_low_price) / last_low_price

    df['open_vec'] = pd.Series(price_vec_matrix[0], index=df.index)
    df['close_vec'] = pd.Series(price_vec_matrix[1], index=df.index)
    df['high_vec'] = pd.Series(price_vec_matrix[2], index=df.index)
    df['low_vec'] = pd.Series(price_vec_matrix[3], index=df.index)

    df['open_change'] = pd.Series(price_vec_matrix[4], index=df.index)
    df['close_change'] = pd.Series(price_vec_matrix[5], index=df.index)
    df['high_change'] = pd.Series(price_vec_matrix[6], index=df.index)
    df['low_change'] = pd.Series(price_vec_matrix[7], index=df.index)

    return df
