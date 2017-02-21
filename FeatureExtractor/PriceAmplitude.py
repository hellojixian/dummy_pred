import pandas as pd


def calculate(df):
    # 计算价格振幅
    # price_amplitude = ( high - low ) / last_close
    # 数值范围 0 - 0.2
    price_change_series = []
    idx_high = df.columns.get_loc('high')
    idx_low = df.columns.get_loc('low')
    idx_close = df.columns.get_loc('close')
    for i in range(0, df.shape[0]):
        last_close = df.iloc[i - 1, idx_close]
        open_price_change = (df.iloc[i, idx_high] - df.iloc[i, idx_low]) / last_close
        price_change_series.append(open_price_change)
    df['amplitude'] = pd.Series(price_change_series, index=df.index)
    return df
