import pandas as pd


def calculate(df):
    # 计算价格振幅
    # price_amplitude = ( high - low ) / last_close
    # 数值范围 0 - 0.2
    price_change_series = []
    price_amp_maxs = []
    price_amp_maxb = []
    idx_high = df.columns.get_loc('high')
    idx_low = df.columns.get_loc('low')
    idx_close = df.columns.get_loc('close')
    idx_open = df.columns.get_loc('open')
    for i in range(0, df.shape[0]):
        last_close = df.iloc[i - 1, idx_close]
        last_high = df.iloc[i - 1, idx_high]
        last_low = df.iloc[i - 1, idx_low]
        high = df.iloc[i, idx_high]
        low = df.iloc[i, idx_low]
        open = df.iloc[i, idx_open]
        open_price_change = (df.iloc[i, idx_high] - df.iloc[i, idx_low]) / last_close
        price_change_series.append(open_price_change)

        maxb = (high-open) / open
        maxs = (open - low) / open
        price_amp_maxb.append(maxb)
        price_amp_maxs.append(maxs)

    df['amplitude'] = pd.Series(price_change_series, index=df.index)
    df['amplitude_maxs'] = pd.Series(price_amp_maxs, index=df.index)
    df['amplitude_maxb'] = pd.Series(price_amp_maxb, index=df.index)
    return df
