import config, warnings, datetime
import pandas as pd
import numpy as np
from FeatureExtractor import PriceAmplitude, PriceVec, PriceChange, CCI, PriceMA, VolMA, Turnover
from sqlalchemy.orm import sessionmaker

RAW_TABLE_NAME = 'raw_stock_trading_5min'
RAW_DAILY_TABLE_NAME = 'raw_stock_trading_daily'

PRICE_MAX, PRICE_MIN = 0, 0
DATE_OFFSET = 1  # 需要多查询几天的日期来生成数据
DAILY_DF = None
limit = ""


def _get_shifted_startdate(stock_code, startdate):
    session = sessionmaker()
    session.configure(bind=config.DB_CONN)
    s = session()

    rs = s.execute(
        "SELECT `date` "
        "FROM {0} "
        "WHERE `code`='{1}' AND `date`='{2}' "
        "LIMIT 0,1".format(
            RAW_DAILY_TABLE_NAME, stock_code, startdate
        )
    )
    data = rs.fetchone()
    if data is None:
        raise RuntimeError('Start date is not a valid trading date {0} {1}'.format(stock_code, startdate))

    rs = s.execute(
        "SELECT `date` "
        "FROM {0} "
        "WHERE `code`='{1}' AND `date`<'{2}' "
        "ORDER BY `date` DESC "
        "LIMIT {3},1".format(
            RAW_DAILY_TABLE_NAME, stock_code, startdate, (DATE_OFFSET - 1)
        )
    )
    data = rs.fetchone()
    new_date = data[0]
    s.close()
    return new_date


def prepare_data(stock_code, startdate, enddate):
    global PRICE_MAX, PRICE_MIN
    global DAILY_DF
    session = sessionmaker()
    session.configure(bind=config.DB_CONN)
    s = session()

    startdate = _get_shifted_startdate(stock_code, startdate)

    rs = s.execute(
        "SELECT `date`, ROUND((`traded_market_value`/`close`)) as total_vol "
        "FROM {0} "
        "WHERE `code`='{1}' AND `date`>='{2}' AND `date`<='{3}' "
        "ORDER BY `date` ASC".format(
            RAW_DAILY_TABLE_NAME, stock_code, startdate, enddate) + limit)
    daily_df = pd.DataFrame(rs.fetchall())
    daily_df.columns = ['date', 'total_vol']
    DAILY_DF = daily_df.set_index(['date'], drop=True)

    rs = s.execute(
        "SELECT MIN(close) as min, MAX(close) as max "
        "FROM {0} "
        "WHERE `code`='{1}' AND `date`>='{2}' AND `date`<='{3}' "
        "ORDER BY `date` ASC".format(
            RAW_DAILY_TABLE_NAME, stock_code, startdate, enddate))
    PRICE_MIN, PRICE_MAX = rs.fetchone()

    rs = s.execute(
        "SELECT * "
        "FROM {0} "
        "WHERE `code`='{1}' AND `time`>='{2}' AND `time`<='{3}' "
        "ORDER BY time ASC".format(
            RAW_TABLE_NAME, stock_code, startdate, enddate) + limit)
    df = pd.DataFrame(rs.fetchall())
    df.columns = ['code', 'time', 'open', 'high', 'low', 'close', 'vol', 'amount', 'count']
    df = df.set_index(['time'], drop=True)
    df = df.drop(labels='code', axis=1)
    df['date'] = [time.date() for time in df.index.tolist()]
    s.close()
    return df


def feature_extraction(df):
    global DAILY_DF
    if 'open_change' not in df.columns:
        df = PriceVec.calculate(df)

    if 'ma5' not in df.columns:
        df = PriceMA.calculate(df)

    if 'v_ma5' not in df.columns:
        df = VolMA.calculate(df)

    if 'change' not in df.columns:
        df = PriceChange.calculate(df)

    if 'amplitude' not in df.columns:
        df = PriceAmplitude.calculate(df)

    if 'cci' not in df.columns:
        df = CCI.calculate(df)

    if 'turnover' not in df.columns:
        df = Turnover.calculate(df, DAILY_DF)

    df = df.dropna(how='any')
    return df


def feature_select(df):
    df = df[["date",
             "open_change", "high_change", "low_change", "close_change",
             "close", "ma5", "ma10", "ma20", "ma30",
             "vol", "v_ma5", "v_ma10", "v_ma20", "v_ma30",
             "change", "amplitude",
             "count", "turnover"]]
    return df


def feature_scaling(df):
    # 价格缩放比
    # 成交量缩放比
    # 振幅/涨幅缩放比
    # 换手率缩放比
    amplitude_scale_rate = 100
    vol_scale_rate = 0.001

    price_min = np.ceil(PRICE_MIN * 0.7)
    price_max = np.ceil(PRICE_MAX * 1.3)

    df[['open_change']] *= amplitude_scale_rate
    df[['high_change']] *= amplitude_scale_rate
    df[['low_change']] *= amplitude_scale_rate
    df[['close_change']] *= amplitude_scale_rate
    df[['change']] *= amplitude_scale_rate
    df[['amplitude']] *= amplitude_scale_rate
    df[['turnover']] *= amplitude_scale_rate

    df[['close']] = (df[['close']] - price_min) / (price_max - price_min)
    df[['ma5']] = (df[['ma5']] - price_min) / (price_max - price_min)
    df[['ma10']] = (df[['ma10']] - price_min) / (price_max - price_min)
    df[['ma20']] = (df[['ma20']] - price_min) / (price_max - price_min)
    df[['ma30']] = (df[['ma30']] - price_min) / (price_max - price_min)

    df[['count']] *= vol_scale_rate
    df[['vol']] *= vol_scale_rate
    df[['v_ma5']] *= vol_scale_rate
    df[['v_ma10']] *= vol_scale_rate
    df[['v_ma20']] *= vol_scale_rate
    df[['v_ma30']] *= vol_scale_rate

    return df


def feature_reshaping(df):
    truncate_index = 36  # data until 14:00pm
    grouped = df.groupby(['date'])

    data_set = []
    index_set = []
    for d, values in grouped:
        values = values.drop(labels='date', axis=1).values
        if len(values) >= truncate_index:
            values = values[:truncate_index]
            data_set.append(values)
            index_set.append(d)

    data_set = np.dstack(data_set)
    data_set = np.swapaxes(data_set, 0, 2)
    data_set = np.swapaxes(data_set, 1, 2)

    return [index_set, data_set]


def prepare_result(index, stock_code, startdate, enddate):
    session = sessionmaker()
    session.configure(bind=config.DB_CONN)
    s = session()

    sql = "SELECT `date`, `open`,`close` " \
          "FROM {0} " \
          "WHERE `code`='{1}' AND `date`>='{2}' AND `date`<='{3}' " \
          "ORDER BY `date` ASC".format(
        RAW_DAILY_TABLE_NAME, stock_code, startdate, enddate) + limit
    rs = s.execute(sql)

    daily_df = pd.DataFrame(rs.fetchall())
    daily_df.columns = ['date', 'open', 'close']
    daily_df = daily_df.set_index(['date'], drop=True)

    result = []
    for date_index in index:
        try:
            date_index = date_index.date()
        except Exception:
            pass
        open = daily_df.loc[date_index, 'open']
        last_date_index = daily_df.index[daily_df.index.get_loc(date_index) - 1]
        last_close = daily_df.loc[last_date_index, 'close']
        change = (open - last_close) / last_close * 100
        r = None
        if change >= 0.5:
            r = [0, 0, 1]  # 上涨
        elif change <= -0.5:
            r = [1, 0, 0]  # 下跌
        else:
            r = [0, 1, 0]  # 平
        result.append(r)
    result = np.asanyarray(result)
    s.close()
    return result
