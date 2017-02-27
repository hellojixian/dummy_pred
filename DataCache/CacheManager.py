import Common.config as config
import warnings, datetime, os
from sqlalchemy.orm import sessionmaker
import pandas as pd
import numpy as np

CACHE_DIR = 'CacheRoot'


class CacheManager:
    _backend = 'csv'
    _cache_table_name = None
    _s = None

    def __init__(self, cache_table_name):
        self._cache_table_name = cache_table_name
        if self._backend == 'mysql':
            session = sessionmaker()
            session.configure(bind=config.DB_CONN)
            self._s = session()
        else:
            pass
        return

    def __del__(self):
        if self._backend == 'mysql':
            self._s.close()
        else:
            pass
        return

    def has_cached_data(self):
        r = False
        if self._backend == 'mysql':
            rs = self._s.execute("SHOW TABLES LIKE '{0}';".format(self._cache_table_name))
            if len(rs.fetchall()):
                r = True
        else:
            path = os.path.join(CACHE_DIR, self._cache_table_name + '.csv')
            r = os.path.isfile(path)
        return r

    def load_cached_data(self):
        if self._backend == 'mysql':
            df = pd.read_sql_table(table_name=self._cache_table_name, con=config.DB_CONN, index_col='time')
        else:
            df = pd.read_csv(os.path.join(CACHE_DIR, self._cache_table_name + '.csv'), index_col='time')
            # convert back the datetime type of fields
            df['date'] = pd.to_datetime(df['date'])
            df.index = pd.to_datetime(df.index)
        return df

    def cache_data(self, data):
        if self._backend == 'mysql':
            data.to_sql(name=self._cache_table_name, con=config.DB_CONN, if_exists="replace", index=True)
        else:
            data.to_csv(path_or_buf=os.path.join(CACHE_DIR, self._cache_table_name + '.csv'), index=True)
        return data
