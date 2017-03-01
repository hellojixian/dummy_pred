#!/usr/bin/env python3
'''
每天收盘以后运行一次，通过Tushare的数据来更新每日数据表
主要获取的内容是为了计算每只股票在分钟线上的换手率参考指标
'''

import os, sys, datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

import DataLoader.TushareDailyDataLoader as DataLoader

if __name__ == "__main__":
    print("\n\nLoading daily data from tushare: \n\n")
    DataLoader.load_data(datetime.date(2017, 2, 14))
