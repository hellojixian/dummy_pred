from datetime import datetime, timedelta, time
from sqlalchemy.orm import sessionmaker
import os, sys, random
import Common.config as config
import numpy as np
import pandas as pd
import h5py

NAME = "DailyFullMarket"
TABLE_NAME_DAILY = "raw_stock_trading_daily"
TABLE_NAME_5MIN_SCALED = "feature_scaled_stock_trading_5min"
TABLE_NAME_5MIN_RESULT = "result_stock_trading_5min"
TABLE_NAME_5MIN_EXTRACTED = "feature_extracted_stock_trading_5min"


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
            length = 48

        return start_time, end_time, length

    def fetch_dataset(self, slice_type):
        if self._dataset is not None:
            return self._dataset

        # if has cache then load from cache
        cache_key = NAME + '-' + slice_type + '-' + str(self.start_date) + '-' + str(self.end_date)
        cache_file = os.path.join(config.CACHE_DIR, cache_key + '.h5')
        if os.path.isfile(cache_file):
            # load cache
            h5f = h5py.File(cache_file, 'r', driver='core')
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
            print(" " * 100 + "\r", end="")
            print(">> processing ... {}%\t[{}/{}] \t\t \r".format(
                round((i / result_count) * 100, 2), index + 1, result_count,
            ), end="")
            sys.stdout.flush()
            code = record['code']
            date = datetime.strptime(str(record['date']), "%Y-%m-%d")

            start_time, end_time, length = self._data_slice(date, slice_type)
            rs = self.db.execute(
                "SELECT * "
                "FROM {} "
                "WHERE `code`='{}' AND `time`>'{}' AND `time`<'{}' "
                "ORDER BY `time` ASC".format(
                    TABLE_NAME_5MIN_SCALED, code, start_time, end_time))
            df = pd.DataFrame(rs.fetchall())
            if df.empty:
                print("code: {} {} is missing training data !!  -  deleted the result and data               ".format(
                    code, date))
                self.db.execute(
                    "DELETE "
                    "FROM {} "
                    "WHERE `code`='{}' AND `date`='{}' "
                    "".format(
                        TABLE_NAME_5MIN_RESULT, code, date))
                self.db.commit()

                self.db.execute(
                    "DELETE "
                    "FROM {} "
                    "WHERE `code`='{}' AND `time`>'{}' AND `time`<'{}' "
                    "".format(
                        TABLE_NAME_5MIN_EXTRACTED, code, date, date + timedelta(days=1)))
                self.db.commit()

                self.db.execute(
                    "DELETE "
                    "FROM {} "
                    "WHERE `code`='{}' AND `time`>'{}' AND `time`<'{}' "
                    "".format(
                        TABLE_NAME_5MIN_SCALED, code, date, date + timedelta(days=1)))
                self.db.commit()
                continue
            df = df.drop(0, axis=1)
            df = df.drop(1, axis=1)

            # verify data length. if not then drop it
            if df.shape[0] == length:
                dataset.append(df)
            else:
                print("code: {} {} is damaged!!  -  deleted the data                ".format(code, date))
                self.db.execute(
                    "DELETE "
                    "FROM {} "
                    "WHERE `code`='{}' AND `time`>'{}' AND `time`<'{}' "
                    "".format(
                        TABLE_NAME_5MIN_SCALED, code, date, date + timedelta(days=1)))
                self.db.commit()

        print("")

        dataset = np.dstack(dataset)
        dataset = np.rollaxis(dataset, -1)

        self._dataset = dataset
        h5f = h5py.File(cache_file, 'w', driver='core')
        h5f.create_dataset('data', data=self._dataset)
        h5f.close()
        return self._dataset

    def fetch_resultset(self, columns=[], cond="True"):
        if self._resultset is not None:
            return self._resultset

        cache_key = NAME + '-' + '_'.join(columns) + str(self.start_date) + '-' + str(self.end_date)
        cache_file = os.path.join(config.CACHE_DIR, cache_key + '-result.csv')
        if os.path.isfile(cache_file):
            df = pd.read_csv(cache_file)
            # df.columns = columns
            self._resultset = df
            return self._resultset

        shifted_start_date = self._get_shifted_startdate()
        columns_list = ['code', 'date'] + columns
        columns_str = '`' + "`,`".join(columns_list) + '`'

        rs = self.db.execute(
            "SELECT {} "
            "FROM {} "
            "WHERE `date`>='{}' AND `date`<='{}' AND ( {} )"
            "ORDER BY `date` ASC".format(
                columns_str, TABLE_NAME_5MIN_RESULT, shifted_start_date, self.end_date, cond))
        df = pd.DataFrame(rs.fetchall())
        df.columns = columns_list
        df.to_csv(path_or_buf=cache_file, index=False)

        self._resultset = df
        return self._resultset

    def balance_result(self, column, low, high, step, samples):
        if self._resultset is None:
            return

        results = self._resultset
        balanced_results = []
        dist = np.arange(low, high + step, step)
        for i in range(len(dist) - 1):
            low = dist[i]
            high = dist[i + 1]
            test = results.query("{0}>{1} and {0}<{2}".format(column, low, high))
            if test.shape[0] > samples:
                random_index = random.sample(test.index.tolist(), samples)
                balanced_results.append(test.loc[random_index])
            else:
                balanced_results.append(test)
        balanced_results = np.vstack(balanced_results)
        balanced_results = pd.DataFrame(balanced_results)
        balanced_results.columns = results.columns

        self._resultset = balanced_results
        return self._resultset

    def balance_dataset(self, dataset, low, high, step, validation_samples, test_samples):
        results = pd.DataFrame(dataset[0])
        data = dataset[1]
        dist = np.arange(low, high + step, step)
        training_set, validation_set, test_set = [[], []], [[], []], [[], []]
        results.columns = ['value']

        for i in range(len(dist) - 1):
            low = dist[i]
            high = dist[i + 1]

            result_piece = results.query("{0}>{1} and {0}<{2}".format('value', low, high))
            data_piece = data[result_piece.index]

            training_set[0].append(result_piece[:(0 - validation_samples - test_samples)])
            training_set[1].append(data_piece[:(0 - validation_samples - test_samples)])
            validation_set[0].append(result_piece[(0 - validation_samples - test_samples):(0 - test_samples)])
            validation_set[1].append(data_piece[(0 - validation_samples - test_samples):(0 - test_samples)])
            test_set[0].append(result_piece[(0 - test_samples):])
            test_set[1].append(data_piece[(0 - test_samples):])

        training_set[0] = np.vstack(training_set[0])
        validation_set[0] = np.vstack(validation_set[0])
        test_set[0] = np.vstack(test_set[0])

        training_set[1] = np.vstack(training_set[1])
        validation_set[1] = np.vstack(validation_set[1])
        test_set[1] = np.vstack(test_set[1])

        # print("-" * 20)
        # print(training_set[0].shape)
        # print(training_set[1].shape)
        #
        # print(validation_set[0].shape)
        # print(validation_set[1].shape)
        #
        # print(test_set[0].shape)
        # print(test_set[1].shape)

        training_result = training_set[0]
        training_data = training_set[1]
        validation_result = validation_set[0]
        validation_data = validation_set[1]
        test_result = test_set[0]
        test_data = test_set[1]

        return [training_data, training_result], \
               [validation_data, validation_result], \
               [test_data, test_result]
