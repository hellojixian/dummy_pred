import pandas as pd
import Common.config as config
import os
import tushare as ts
from datetime import datetime
from sqlalchemy.orm import sessionmaker

DB = None


def load_data(start_date=datetime.now().date()):
    stock_list = _get_stock_list()

    global DB
    session = sessionmaker()
    session.configure(bind=config.DB_CONN)
    DB = session()
    print("Fetching stock data since {}".format(start_date))
    i = 0
    for index, record in stock_list.iterrows():
        i += 1
        code = str(record['code'])
        if code[:3] != '900' and code[:3] != '200' and len(code) == 6:
            print("[{}/{}]\t Stock: {} \t".format(i, stock_list.shape[0], code), end='')
            stock_data = ts.get_hist_data(code, ktype="D", start=str(start_date), retry_count=10, pause=2)
            print("Data: {} ".format(stock_data.shape[0]), end="")
            print("[", end="")
            _save_record(code, stock_data)
            print("]")
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
    print("{} stocks found".format(stock_list.shape[0]))
    return stock_list


def _save_record(code, data):
    if code[:2] == '60':
        code = "sh" + code
    elif code[:2] == '00' or code[:2] == '30':
        code = "sz" + code
    else:
        return

    for date, record in data.iterrows():
        columns = ["date", "open", "high", "close", "low", "p_change", "volume", "turnover"]
        record = record[columns]
        traded_market_value = record['volume'] / record['turnover'] * record['close'] * 100
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
    pass
