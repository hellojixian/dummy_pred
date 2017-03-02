from datetime import timedelta
from sqlalchemy.orm import sessionmaker
import Common.config as config
import time, sys
import numpy as np

TABLE_NAME_5MIN = "raw_stock_trading_5min"
TABLE_NAME_DAILY = "raw_stock_trading_daily"


class Transform5M:
    def __init__(self):
        pass


def process_date_range(start_date, end_date):
    date_diff = end_date - start_date
    session = sessionmaker()
    session.configure(bind=config.DB_CONN)
    s = session()
    for diff in range(date_diff.days):
        delta = timedelta(days=diff)
        the_date = start_date + delta
        print("Transforming data: {}\t".format(the_date), end="")
        sql = "SELECT `code` FROM `{}` WHERE `date`='{}' GROUP BY `code`".format(
            TABLE_NAME_DAILY, the_date
        )
        rs = s.execute(sql)
        df = rs.fetchall()

        stock_count = len(df)
        print(" - {} stocks found".format(stock_count))
        if len(df) == 0:
            continue
        for i in range(stock_count):
            code = df[i][0]
            print(">> Processing ... {}%\t\tCode: {} [{}/{}]  \r"
                  .format(np.round((i + 1) / stock_count*100,1), code, i + 1, stock_count), end="")
            sys.stdout.flush()
            # print(code)
            # 每股的处理代码在这里
            # t5m = Transform5M(code, the_date)

            time.sleep(0.005)
            if (i+1) == stock_count:
                print(" " * 100 + "\r", end="")
                print(">> Processing ... 100%\t[ DONE ]  \r", end="")
                sys.stdout.flush()
                time.sleep(0.5)
                print(" " * 100 + "\r", end="")
                sys.stdout.flush()


    s.close()
    return
