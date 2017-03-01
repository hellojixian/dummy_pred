import pandas as pd
import Common.config as config
import os


def load_data_to_db():
    for cur, _dirs, files in os.walk(config.STOCK_DATA_PATH):
        print(cur)
        for f in files:
            file_path = os.path.join(cur, f);
            print(file_path)
            if file_path.endswith('csv'):
                print("Loading data file {0}".format(f))
                _load_stock_data_file(file_path)

    for cur, _dirs, files in os.walk(config.INDEX_DATA_PATH):
        print(cur)
        for f in files:
            file_path = os.path.join(cur, f);
            print(file_path)
            if file_path.endswith('csv'):
                print("Loading index file {0}".format(f))
                _load_index_data_file(file_path)


def _load_stock_data_file(file):
    f_name = os.path.basename(file)
    df = pd.read_csv(file)
    print("{} contains: {} records".format(f_name, df.shape[0]))
    print("Loading: [ ", end="")
    for i in range(df.shape[0]):
        rec = df.iloc[i:i + 1]
        try:
            rec.to_sql(name='raw_stock_trading_daily', con=config.DB_CONN, if_exists="append", index=False)
            print(".", end="")
        except Exception:
            break
    print(" ]")
    print('{} loaded'.format(f_name))
    return


def _load_index_data_file(file):
    f_name = os.path.basename(file)
    df = pd.read_csv(file)
    df = df.rename(index=str, columns={"index_code": "code"})
    print("{} contains: {} records".format(f_name, df.shape[0]))
    print("Loading: [ ", end="")
    for i in range(df.shape[0]):
        rec = df.iloc[i:i + 1]
        try:
            rec.to_sql(name='raw_stock_index_daily', con=config.DB_CONN, if_exists="append", index=False)
            print(".", end="")
        except Exception:
            break
    print(" ]")
    print('{} loaded'.format(f_name))
    return
