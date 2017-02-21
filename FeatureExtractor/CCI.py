import pandas as pd


def calculate(df):
    # http://baike.baidu.com/item/%E9%A1%BA%E5%8A%BF%E6%8C%87%E6%A0%87/976711?fromtitle=CCI&fromid=8862685&type=syn
    # TP =（high + low + close）/ 3
    # MA(n) = sum(close,n)/n
    # MD(n) = sum(MA(n) - close) / n
    # cci(n) = （TP－MA(n)）/ MD(n) / 0.015

    return df