import pandas as pd
import numpy as np


def calculate(df):
    # 1．24 天以来凡是股价上涨那一天的成交量都称为AV，将24天内的AV总和相加后称为AVS。
    # 2．24 天以来凡是股价下跌那一天的成交量都称为BV，将24天内的BV总和相加后称为BVS。
    # 3．24 天以来凡是股价不涨不跌，则那一天的成交量都称为CV，将24天内的CV总和相加后称为CVS。
    # 4． 24 天开始计算：
    # VR =（AVS + 1/2 * CVS） / （BVS + 1/2 * CVS）

    n = 24
    vr_list = []
    for i in range(0, df.shape[0]):
        mx = df.iloc[i - n:i, [df.columns.get_loc('change'), df.columns.get_loc('vol')]]
        avs = float(np.sum(mx.query("change>0").drop('change', axis=1)))
        bvs = float(np.sum(mx.query("change<0").drop('change', axis=1)))
        cvs = float(np.sum(mx.query("change==0").drop('change', axis=1)))
        vr = 0
        if (bvs + 0.5 * cvs) != 0:
            vr = (avs + 0.5 * cvs) / (bvs + 0.5 * cvs)
        vr_list.append(vr)

    df['vr'] = pd.Series(vr_list, index=df.index)
    return df
