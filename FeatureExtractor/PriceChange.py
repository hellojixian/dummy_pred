import pandas as pd


def calculate(df):
    # 计算价格涨幅
    # open_price_change = ( close - last_close ) / last_close
    # 数值范围 +0.1 / - 0.1
    price_change_list = []
    idx_close = df.columns.get_loc('close')
    for i in range(0, df.shape[0]):
        last_close = df.iloc[i - 1, 3]
        open_price_change = (df.iloc[i, idx_close] - last_close) / last_close
        price_change_list.append(open_price_change)
    df['change'] = pd.Series(price_change_list, index=df.index)
    return df