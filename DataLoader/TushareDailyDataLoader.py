import pandas as pd
import numpy as np
import Common.config as config
import os, sys, time
import tushare as ts
from datetime import datetime
from datetime import timedelta
from sqlalchemy.orm import sessionmaker

DB = None


def load_data(start_date=datetime.now().date()):
    stock_list = _get_stock_list()

    global DB
    session = sessionmaker()
    session.configure(bind=config.DB_CONN)
    DB = session()
    print("Fetching stock data since {}".format(start_date))
    stock_list = stock_list['code'].tolist()
    stocks_str = []
    for code in stock_list:
        code = str(code)
        if len(code) == 6 \
                and (code[:2] == '60' \
                             or code[:2] == '00' \
                             or code[:2] == '30'):
            stocks_str.append(code)
    # 两层循环嵌套，第一层按日期循环 第二层按批次轮训
    today = datetime.now().date()
    date_diff = today - start_date
    step = 100
    for diff in range(date_diff.days + 1):
        delta = timedelta(days=diff)
        the_date = start_date + delta
        print("Processing stock data for {} ".format(the_date))
        for i in range(0, len(stocks_str), step):
            stocks = stocks_str[i:i + step]
            print("Batch {}/{}\t Data:{}  [ Fetching ... ]".format(i, len(stocks_str), len(stocks)), end="")
            sys.stdout.flush()
            print("", end="")
            # print(stocks)
            sys.stdout.flush()
            stock_data = ts.get_hists(stocks,
                                      ktype="D",
                                      start=str(the_date),
                                      end=str(the_date),
                                      retry_count=10,
                                      pause=2)
            print("                                 \r", end="")
            sys.stdout.flush()
            print("Batch {}/{}\t Data:{}  [ Done ]".format(i, len(stocks_str), len(stocks)), end="")
            sys.stdout.flush()
            time.sleep(1)
            print("                                  \r", end="")
            sys.stdout.flush()

            print("Batch {}/{}\t Data:{}  [ ".format(i, len(stocks_str), len(stocks)), end="")
            if stock_data is not None:
                _save_record(the_date, stock_data)
            print(" ]")

    DB.close()
    pass


def _get_stock_list():
    cachedFile = os.path.join(config.CACHE_DIR, "stock_list.csv")
    if os.path.isfile(cachedFile):
        print("Fetching stock list from Cache")
        stock_list = pd.read_csv(cachedFile)
    else:
        print("Fetching stock list from TuShare")
        stock_list = ts.get_industry_classified()
        stock_list.to_csv(path_or_buf=cachedFile, index=False)
    print("{} stocks found in total".format(stock_list.shape[0]))
    return stock_list


def _save_record(date, data):
    # print(record)
    # print(data)
    for i, record in data.iterrows():
        columns = ["code", "date", "open", "high", "close", "low", "p_change", "volume", "turnover"]
        record = record[columns]
        # 转换统一数据缩放比例
        code = str(record['code'])
        if code[:2] == '60':
            code = "sh" + code
        elif code[:2] == '00' or code[:2] == '30':
            code = "sz" + code
        else:
            continue
        record['p_change'] /= 100
        record['turnover'] /= 100
        record['volume'] *= 100
        record['volume'] = int(record['volume'])
        record['turnover'] = np.round(record['turnover'], 6)
        traded_market_value = record['volume'] / record['turnover'] * record['close']
        traded_market_value = np.round(traded_market_value, 2)
        sql = "INSERT INTO raw_stock_trading_daily " \
              "(`code`, `date`, `open`, `high`, `low`, `close`, `change`, `volume`, `traded_market_value`, `turnover`) " \
              "VALUES ('{}','{}','{}','{}','{}','{}','{}','{}','{}','{}'); ".format(
            code, date, record['open'], record['high'], record['low'], record['close'], record['p_change'],
            record['volume'], traded_market_value, record['turnover']
        )
        try:
            DB.execute(sql)
            DB.commit()
            print('.', end='')
        except Exception:
            print(':', end='')
        sys.stdout.flush()
    pass
