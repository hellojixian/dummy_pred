#!/usr/bin/env python3
'''
转换5分钟K线的原始数据
默认先查找当天数据库的所有股票，然后按每日便利每只股票
'''

import os, sys, datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from DataTransform.Transform5M import process_date_range

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("{0} start_date end_date".format(sys.argv[0]))
        exit(0)

    start_date = datetime.datetime.strptime(str(sys.argv[1]), "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(str(sys.argv[2]), "%Y-%m-%d").date()
    process_date_range(start_date, end_date)
