import pandas as pd
import Common.config as config
from glob import glob


def load_data_to_db():
    files = glob(config.STOCK_DATA_PATH + '/*.csv')
    for file in files:
        _load_stock_data_file(file)

    files = glob(config.INDEX_DATA_PATH + '/*.csv')
    for file in files:
        _load_index_data_file(file)


def _load_stock_data_file(file):
    df = pd.read_csv(file)
    df.to_sql(name='raw_stock_trading_daily', con=config.DB_CONN, if_exists="append", index=False)
    print('Stock File {} loaded'.format(file))


def _load_index_data_file(file):
    df = pd.read_csv(file)
    df = df.rename(index=str, columns={"index_code": "code"})
    df.to_sql(name='raw_stock_index_daily', con=config.DB_CONN, if_exists="append", index=False)
    print('Index File {} loaded'.format(file))
