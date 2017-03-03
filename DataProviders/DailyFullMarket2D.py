from datetime import datetime, timedelta, time
from sqlalchemy.orm import sessionmaker
import os
import Common.config as config
import numpy as np
import pandas as pd
import h5py

NAME = "DailyFullMarket"
TABLE_NAME_DAILY = "raw_stock_trading_daily"
TABLE_NAME_5MIN_SCALED = "feature_scaled_stock_trading_5min"
TABLE_NAME_5MIN_RESULT = "result_stock_trading_5min"


class DailyFullMarket2D:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self._shifted_date = None
        self._dataset = None
        self._resultset = None
        session = sessionmaker()
        session.configure(bind=config.DB_CONN)
        self.db = session()
        return

    def __del__(self):
        self.db.close()
        return

    def _get_shifted_startdate(self):
        if self._shifted_date is not None:
            return self._shifted_date

        sql = "SELECT `date` " \
              "FROM {0} " \
              "WHERE `date`>='{1}' " \
              "ORDER BY `date` ASC " \
              "LIMIT 0,1".format(
            TABLE_NAME_DAILY, self.start_date
        )
        rs = self.db.execute(sql)
        data = rs.fetchone()
        if data is None:
            raise RuntimeError('No more trading date at or after {1}'.format(self.start_date))
            return None

        rs = self.db.execute(
            "SELECT `date` "
            "FROM {0} "
            "WHERE `date`<'{1}' "
            "ORDER BY `date` DESC "
            "LIMIT 0,1".format(
                TABLE_NAME_DAILY, self.start_date
            )
        )
        data = rs.fetchone()
        shifted_date = data[0]
        self._shifted_date = shifted_date

        return shifted_date

    def _data_slice(self, date, type):
        start_time, end_time = date, date

        if type == 'today_full':
            end_time = date + timedelta(days=1)

        return start_time, end_time

    def fetch_dataset(self, slice_type):
        if self._dataset is not None:
            return self._dataset

        # if has cache then load from cache
        cache_key = NAME + '-' + slice_type + '-' + str(self.start_date) + '-' + str(self.end_date)
        cache_file = os.path.join(config.CACHE_DIR, cache_key + '.h5')
        if os.path.isfile(cache_file):
            # load cache
            h5f = h5py.File(cache_file, 'r')
            self._dataset = h5f['data'][:]
            h5f.close()
            return self._dataset

        results = self.fetch_resultset()
        result_count = len(results)
        print("Fetching dataset: {} records".format(result_count))

        dataset = []
        i = 0
        for index, record in results.iterrows():
            i += 1
            # if i > 10: break
            print(">> processing ... {}%\t[{}/{}] \t\t \r".format(
                np.round((i / result_count) * 100, 2), index + 1, result_count,
            ), end="")
            code = record['code']
            date = record['date']
            start_time, end_time = self._data_slice(date, slice_type)
            rs = self.db.execute(
                "SELECT * "
                "FROM {} "
                "WHERE `code`='{}' AND `time`>'{}' AND `time`<'{}' "
                "ORDER BY `time` ASC".format(
                    TABLE_NAME_5MIN_SCALED, code, start_time, end_time))
            df = pd.DataFrame(rs.fetchall())
            df = df.drop(0, axis=1)
            df = df.drop(1, axis=1)
            dataset.append(df)

        print("")

        dataset = np.dstack(dataset)
        dataset = np.rollaxis(dataset, -1)

        self._dataset = dataset
        h5f = h5py.File(cache_file, 'w')
        h5f.create_dataset('data', data=self._dataset)
        h5f.close()
        return self._dataset

    def fetch_resultset(self, columns=[]):

        shifted_start_date = self._get_shifted_startdate()
        columns = ['code', 'date'] + columns
        columns_str = '`' + "`,`".join(columns) + '`'

        rs = self.db.execute(
            "SELECT {} "
            "FROM {} "
            "WHERE `date`>='{}' AND `date`<='{}' "
            "ORDER BY `date` ASC".format(
                columns_str, TABLE_NAME_5MIN_RESULT, shifted_start_date, self.end_date))
        df = pd.DataFrame(rs.fetchall())
        df.columns = columns

        self._resultset = df
        return self._resultset
