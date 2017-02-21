from sqlalchemy import create_engine
import pandas as pd

STOCK_DATA_PATH = '/Users/jixianwang/stock_data/stock_daily_data'
INDEX_DATA_PATH = '/Users/jixianwang/stock_data/index_daily_data'
MINUTE_DATA_PATH = '/Users/jixianwang/stock_data/minute_daily_data'

MYSQL_CONN = 'mysql://stock_prediction:jixian@127.0.0.1/stock_prediction?charset=utf8'
DB_CONN = create_engine(MYSQL_CONN)


pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 200)