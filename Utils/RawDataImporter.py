#!/usr/bin/env python3
'''
从 minute_daily_data 文件夹中批量导入每日的分钟数据
数据格式是 yucezhe.com 的，程序会自动解压缩并且导入数据库，
如果每天的数据有重复程序会自动跳过
'''

import os,sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

import Common.config as config
import DataLoader.MinutesDataLoader as DataLoader

if __name__ == "__main__":
    print("\n\nImporting data from path: {0} \n\n".format(config.MINUTE_DATA_PATH))
    DataLoader.load_data_to_db()