import pandas as pd


def calculate(df, daily_df):
    # 计算每日的流通股有多少手
    # 查询 raw_stock_trading_daily 获取当天的记录
    # total_vol = traded_market_value / close
    # turnover_rate = volume / total_vol
    # 数值范围 0 - 0.3
    turnover_list = []
    for i in range(0, df.shape[0]):
        date = df.index[i].date()
        total_vol = daily_df.loc[date, 'total_vol']
        turnover = df.loc[df.index[i], 'vol'] / total_vol * 100
        turnover_list.append(turnover)
    df['turnover'] = pd.Series(turnover_list, index=df.index)

    return df
