import pandas as pd
import numpy as np

def calculate(df):
    price_vec_matrix = np.zeros((4, df.shape[0]))
    idx_high = df.columns.get_loc('high')
    idx_low = df.columns.get_loc('low')
    idx_close = df.columns.get_loc('close')
    idx_open = df.columns.get_loc('open')
    for i in range(0, df.shape[0]):
        mean_price = np.mean([df.iloc[i, idx_close],
                              df.iloc[i, idx_open],
                              df.iloc[i, idx_high],
                              df.iloc[i, idx_low]])
        price_vec_matrix[0, i] = (df.iloc[i, idx_open] - mean_price) / mean_price
        price_vec_matrix[1, i] = (df.iloc[i, idx_close] - mean_price) / mean_price
        price_vec_matrix[2, i] = (df.iloc[i, idx_high] - mean_price) / mean_price
        price_vec_matrix[3, i] = (df.iloc[i, idx_low] - mean_price) / mean_price
    df['open_change'] = pd.Series(price_vec_matrix[0], index=df.index)
    df['high_change'] = pd.Series(price_vec_matrix[2], index=df.index)
    df['low_change'] = pd.Series(price_vec_matrix[3], index=df.index)
    df['close_change'] = pd.Series(price_vec_matrix[1], index=df.index)
    return df