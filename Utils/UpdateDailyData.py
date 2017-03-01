#!/usr/bin/env python3
'''
每天收盘以后运行一次，通过Tushare的数据来更新每日数据表
主要获取的内容是为了计算每只股票在分钟线上的换手率参考指标

哎~ 虽然写完了 但是速度慢的像屎一样 可能我在国外的原因
'''

import os, sys, datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

import DataLoader.TushareDailyDataLoader as DataLoader

if __name__ == "__main__":
    if len(sys.argv) == 2:
        start_date = datetime.datetime.strptime(str(sys.argv[1]), "%Y-%m-%d").date()
    else:
        start_date = datetime.datetime.now().date()
    print("Loading daily data from tushare: ")
    DataLoader.load_data(start_date)
