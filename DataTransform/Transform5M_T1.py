import Common.config as config
import warnings, datetime
import pandas as pd
import numpy as np
from FeatureExtractor import PriceAmplitude, PriceVec, PriceChange, \
    CCI, PriceMA, VolMA, Turnover, RSI, KDJ, BIAS, BOLL, ROC, \
    VR, WR, MI, OSCV, DMA, EMV, EXPMA, ARBR, DMI, ASI, MACD, PSY, WVAD
from sqlalchemy.orm import sessionmaker

RAW_TABLE_NAME = 'raw_stock_trading_5min'
RAW_DAILY_TABLE_NAME = 'raw_stock_trading_daily'

PRICE_MAX, PRICE_AVG, PRICE_MIN, STOCK_CODE = 0, 0, 0, 0
VOL_MAX, VOL_AVG, VOL_MIN = 0, 0, 0
COUNT_MAX, COUNT_MIN = 0, 0
LIMIT_SAMPLE_START = '2016-01-01'
LIMIT_SAMPLE_END = '2017-01-01'
DATE_OFFSET = 1  # 需要多查询几天的日期来生成数据
DAILY_DF = None
limit = ""


def _get_shifted_startdate(stock_code, startdate):
    session = sessionmaker()
    session.configure(bind=config.DB_CONN)
    s = session()

    sql = "SELECT `date` " \
          "FROM {0} " \
          "WHERE `code`='{1}' AND `date`>='{2}' ORDER BY `date` ASC " \
          "LIMIT 0,1".format(
        RAW_DAILY_TABLE_NAME, stock_code, startdate
    )
    rs = s.execute(sql)
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


def init(stock_code, start_date):
    global STOCK_CODE, \
        PRICE_MAX, PRICE_AVG, PRICE_MIN, \
        VOL_MAX, VOL_AVG, VOL_MIN, \
        COUNT_MAX, COUNT_MIN, \
        LIMIT_SAMPLE_START, LIMIT_SAMPLE_END

    time_offset = datetime.timedelta(days=30)
    LIMIT_SAMPLE_START = datetime.date(start_date.year, start_date.month, 1) - time_offset * 2
    LIMIT_SAMPLE_END = LIMIT_SAMPLE_START + time_offset
    print("Date sample range: {0} to {1}".format(LIMIT_SAMPLE_START, LIMIT_SAMPLE_END))
    session = sessionmaker()
    session.configure(bind=config.DB_CONN)
    s = session()

    rs = s.execute(
        "SELECT "
        "MIN(vol) as vol_min,  AVG(vol) as vol_avg, MAX(vol) as vol_max, "
        "MIN(close) as vol_min,  AVG(close) as vol_avg, MAX(close) as vol_max, "
        "MIN(count) as count_min, MAX(count) as count_max "
        "FROM {0} "
        "WHERE `code`='{1}' AND `time`>='{2}' AND `time`<='{3}' ".format(
            RAW_TABLE_NAME, stock_code, LIMIT_SAMPLE_START, LIMIT_SAMPLE_END))
    VOL_MIN, VOL_AVG, VOL_MAX, \
    PRICE_MIN, PRICE_AVG, PRICE_MAX, \
    COUNT_MIN, COUNT_MAX = rs.fetchone()

    STOCK_CODE = stock_code
    s.close()
    print("Price: {} - {} - {} \n"
          "Vol: {} - {} - {} \n"
          "Count: {} - {}".format(PRICE_MIN, PRICE_AVG, PRICE_MAX, VOL_MIN, VOL_AVG, VOL_MAX, COUNT_MIN, COUNT_MAX))
    return


def prepare_data(startdate, enddate):
    global PRICE_MAX, PRICE_MIN
    global DAILY_DF
    session = sessionmaker()
    session.configure(bind=config.DB_CONN)
    s = session()

    startdate = _get_shifted_startdate(STOCK_CODE, startdate)

    rs = s.execute(
        "SELECT `date`, ROUND((`traded_market_value`/`close`)) as total_vol "
        "FROM {0} "
        "WHERE `code`='{1}' AND `date`>='{2}' AND `date`<='{3}' "
        "ORDER BY `date` ASC".format(
            RAW_DAILY_TABLE_NAME, STOCK_CODE, startdate, enddate) + limit)
    daily_df = pd.DataFrame(rs.fetchall())
    daily_df.columns = ['date', 'total_vol']
    DAILY_DF = daily_df.set_index(['date'], drop=True)

    rs = s.execute(
        "SELECT * "
        "FROM {0} "
        "WHERE `code`='{1}' AND `time`>='{2}' AND `time`<='{3}' "
        "ORDER BY time ASC".format(
            RAW_TABLE_NAME, STOCK_CODE, startdate, enddate) + limit)
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

    if 'cci_5' not in df.columns:
        df = CCI.calculate(df)

    if 'rsi_6' not in df.columns:
        df = RSI.calculate(df)

    if 'k' not in df.columns:
        df = KDJ.calculate(df)

    if 'bias_5' not in df.columns:
        df = BIAS.calculate(df)

    if 'boll_md' not in df.columns:
        df = BOLL.calculate(df)

    if 'roc_12' not in df.columns:
        df = ROC.calculate(df)

    if 'vr' not in df.columns:
        df = VR.calculate(df)

    if 'turnover' not in df.columns:
        df = Turnover.calculate(df, DAILY_DF)

    if 'wr_5' not in df.columns:
        df = WR.calculate(df)

    if 'mi_5' not in df.columns:
        df = MI.calculate(df)

    if 'oscv' not in df.columns:
        df = OSCV.calculate(df)

    if 'dma_dif' not in df.columns:
        df = DMA.calculate(df)

    if 'emv_emv' not in df.columns:
        df = EMV.calculate(df)

    if 'ema_5' not in df.columns:
        df = EXPMA.calculate(df)

    if 'ar' not in df.columns:
        df = ARBR.calculate(df)

    if 'adx' not in df.columns:
        df = DMI.calculate(df)

    if 'asi_5' not in df.columns:
        df = ASI.calculate(df)

    if 'macd_dif' not in df.columns:
        df = MACD.calculate(df)

    if 'psy' not in df.columns:
        df = PSY.calculate(df)

    if 'wvad' not in df.columns:
        df = WVAD.calculate(df)

    df = df.dropna(how='any')

    # print(df.shape)
    # print(df[36:60])
    return df


def _features():
    features = ["date",
                "open_vec", "high_vec", "low_vec", "close_vec",
                "open_change", "high_change", "low_change", "close_change",
                "close", "ma5", "ma15", "ma25", "ma40",
                "ema_5", "ema_15", "ema_25", "ema_40",
                "boll_up", "boll_md", "boll_dn",
                "turnover", "count",
                "vol", "vr", "v_ma5", "v_ma15", "v_ma25", "v_ma40",
                "cci_5", "cci_15", "cci_30",
                "rsi_6", "rsi_12", "rsi_24",
                "k9", "d9", "j9",
                "bias_5", "bias_10", "bias_30",
                "roc_12", "roc_25",
                "change", "amplitude", "amplitude_maxb", "amplitude_maxs",
                "wr_5", "wr_10", "wr_20",
                "mi_5", "mi_10", "mi_20", "mi_30",
                "oscv",
                "dma_dif", "dma_ama",
                "ar", "br",
                "pdi", "mdi", "adx", "adxr",
                "asi_5", "asi_15", "asi_25", "asi_40",
                "macd_dif", "macd_dea", "macd_bar",
                "psy", "psy_ma",
                "emv_emv", "emv_maemv",
                "wvad", "wvad_ma"
                ]
    return features


def features():
    f_list = _features()
    f_list.remove('date')
    f_list.remove('close')
    return f_list


def feature_select(df):
    df = df[_features()]
    return df


def feature_scaling(df):
    print(df.head(100))
    # 价格缩放比
    # 成交量缩放比
    # 振幅/涨幅缩放比
    # 换手率缩放比
    amplitude_scale_rate = 100
    rsi_scale_rate = 0.01
    oscv_scale_rate = 0.01
    wr_scale_rate = 0.04
    macd_scale_rate = 50

    # price_min = np.ceil(PRICE_MIN * 0.7)
    # price_max = np.ceil(PRICE_MAX * 1.3)

    df[['open_change']] *= amplitude_scale_rate * 1.5
    df[['high_change']] *= amplitude_scale_rate * 1.5
    df[['low_change']] *= amplitude_scale_rate * 1.5
    df[['close_change']] *= amplitude_scale_rate * 1.5
    df[['open_vec']] *= amplitude_scale_rate * 1.5
    df[['high_vec']] *= amplitude_scale_rate * 1.5
    df[['low_vec']] *= amplitude_scale_rate * 1.5
    df[['close_vec']] *= amplitude_scale_rate * 1.5

    # 下面这组数据应该与收盘价来做缩放
    # 否则这么多维度数据数值都非常接近
    # 缩放算法是 scaled = (value - close) * scale_rate_l2
    for index, row in df.iterrows():
        close = df.loc[index, 'close']
        vol = df.loc[index, 'vol']
        if vol == 0:
            vol = 1
        df.loc[index, 'ma5'] = (df.loc[index, 'ma5'] - close) / close * 100 / 2
        df.loc[index, 'ma15'] = (df.loc[index, 'ma15'] - close) / close * 100 / 2
        df.loc[index, 'ma25'] = (df.loc[index, 'ma25'] - close) / close * 100 / 2
        df.loc[index, 'ma40'] = (df.loc[index, 'ma40'] - close) / close * 100 / 2

        df.loc[index, 'ema_5'] = (df.loc[index, 'ema_5'] - close) / close * 100 / 2
        df.loc[index, 'ema_15'] = (df.loc[index, 'ema_15'] - close) / close * 100 / 2
        df.loc[index, 'ema_25'] = (df.loc[index, 'ema_25'] - close) / close * 100 / 2
        df.loc[index, 'ema_40'] = (df.loc[index, 'ema_40'] - close) / close * 100 / 2

        df.loc[index, 'boll_up'] = (df.loc[index, 'boll_up'] - close) / close * 100 / 2
        df.loc[index, 'boll_dn'] = (df.loc[index, 'boll_dn'] - close) / close * 100 / 2
        df.loc[index, 'boll_md'] = (df.loc[index, 'boll_md'] - close) / close * 100 / 2

        df.loc[index, 'v_ma5'] = (df.loc[index, 'v_ma5'] - vol) / vol / 2
        df.loc[index, 'v_ma15'] = (df.loc[index, 'v_ma15'] - vol) / vol / 2
        df.loc[index, 'v_ma25'] = (df.loc[index, 'v_ma25'] - vol) / vol / 2
        df.loc[index, 'v_ma40'] = (df.loc[index, 'v_ma40'] - vol) / vol / 2

        df.loc[index, 'mi_5'] = df.loc[index, 'mi_5'] / close * 80
        df.loc[index, 'mi_10'] = df.loc[index, 'mi_10'] / close * 80
        df.loc[index, 'mi_20'] = df.loc[index, 'mi_20'] / close * 80
        df.loc[index, 'mi_30'] = df.loc[index, 'mi_30'] / close * 80

    # 最后再把价格计算差值

    df[['change']] *= amplitude_scale_rate
    df[['amplitude']] *= amplitude_scale_rate
    df[['amplitude_maxs']] *= amplitude_scale_rate
    df[['amplitude_maxb']] *= amplitude_scale_rate
    df[['turnover']] *= amplitude_scale_rate * 10
    df[['roc_12']] *= amplitude_scale_rate / 2
    df[['roc_25']] *= amplitude_scale_rate / 2

    df[['count']] = (df[['count']] - COUNT_MIN) / (COUNT_MAX - COUNT_MIN)
    df[['vol']] = (df[['vol']] - VOL_MIN) / (VOL_MAX - VOL_MIN) * 6
    df[['vr']] *= 0.5

    df[['cci_5']] *= 0.003
    df[['cci_15']] *= 0.003
    df[['cci_30']] *= 0.003

    df[['rsi_6']] *= rsi_scale_rate
    df[['rsi_12']] *= rsi_scale_rate
    df[['rsi_24']] *= rsi_scale_rate

    df[['k9']] *= rsi_scale_rate
    df[['d9']] *= rsi_scale_rate
    df[['j9']] *= rsi_scale_rate

    df[['wr_5']] *= wr_scale_rate
    df[['wr_10']] *= wr_scale_rate
    df[['wr_20']] *= wr_scale_rate

    df[['oscv']] *= oscv_scale_rate

    df[['dma_dif']] *= (PRICE_MAX - PRICE_MIN) / PRICE_MIN * 8
    df[['dma_ama']] *= (PRICE_MAX - PRICE_MIN) / PRICE_MIN * 8

    df[['ar']] = (df[['ar']] - 100) * 0.01
    df[['br']] = (df[['br']] - 100) * 0.01

    df[['mdi']] *= 2
    df[['pdi']] *= 2
    df[['adx']] *= 0.5
    df[['adxr']] *= 0.5

    df[['asi_5']] *= 1 / PRICE_AVG
    df[['asi_15']] *= 1 / PRICE_AVG
    df[['asi_25']] *= 1 / PRICE_AVG
    df[['asi_40']] *= 1 / PRICE_AVG

    df[['macd_bar']] *= 1 / PRICE_AVG * 100
    df[['macd_dea']] *= 1 / PRICE_AVG * 100
    df[['macd_dif']] *= 1 / PRICE_AVG * 100

    df[['psy']] *= 0.01
    df[['psy_ma']] *= 0.01

    df[['emv_emv']] *= (PRICE_AVG / VOL_AVG)
    df[['emv_maemv']] *= (PRICE_AVG / VOL_AVG)

    df[['wvad']] = (df[['wvad']] - VOL_MIN) / (VOL_MAX - VOL_MIN) * 2.4
    df[['wvad_ma']] = (df[['wvad_ma']] - VOL_MIN) / (VOL_MAX - VOL_MIN) * 2.4

    # df[['close']] = (df[['close']] - price_min) / (price_max - price_min)
    df = df.drop(labels='close', axis=1)

    print(df.head(400))
    print(df.shape)
    print(pd.DataFrame(df.columns[1:]))
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
