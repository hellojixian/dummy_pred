import os, zipfile
import pandas as pd
import config


def load_data_to_db():
    path = os.path.join(config.MINUTE_DATA_PATH)
    for cur, _dirs, files in os.walk(path):
        print(cur)
        for f in files:
            file_path = os.path.join(cur, f);
            print(file_path)
            if file_path.endswith('zip'):
                print("Extracting file {0}".format(file_path))
                _unzip_file(cur, file_path)
            elif file_path.endswith('csv'):
                print("Loading data file {0}".format(file_path))
                _load_data_files(file_path)


def _unzip_file(cur, file):
    zip_ref = zipfile.ZipFile(file)  # create zipfile object
    zip_ref.extractall(cur)  # extract file to dir
    zip_ref.close()  # close file
    os.remove(file)


def _load_data_files(file_path):
    table_name = None
    if file_path.endswith('1min.csv'):
        table_name = "raw_stock_trading_1min"
    elif file_path.endswith(' 5min.csv'):
        table_name = "raw_stock_trading_5min"
    elif file_path.endswith('15min.csv'):
        table_name = "raw_stock_trading_15min"
    elif file_path.endswith("30min.csv"):
        table_name = "raw_stock_trading_30min"

    if table_name:
        df = pd.read_csv(filepath_or_buffer=file_path,
                         skiprows=2,
                         header=0,
                         names=['code', 'time', 'open', 'high', 'low', 'close', 'vol', 'amount', 'count'])
        df.to_sql(name=table_name, con=config.DB_CONN, if_exists="append", index=False)
        print('Stock File {} loaded'.format(file_path))
