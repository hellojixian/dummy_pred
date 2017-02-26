from functools import reduce

# 简单算术移动平均
def MA(vals):
    ret = 0
    for x in vals:
        ret = ret + x
    return ret / len(vals)

# 加权移动平均
# 同花顺和通达信等软件中的SMA
#
# 算法
# Y = (X*M + Y'*(N-M)) / N
# 2种实现，结果相同
# 注：使用SMA函数，历史数据越多越准确，EMA同
def SMA(vals, n, m) :
    # 算法1
    return reduce(lambda x, y: ((n - m) * x + y * m) / n, vals)

    '''
    ret = vals[0]
    for x in vals:
        ret = (x * m + ret * (n - m)) / n
    return ret
    '''

# 指数平滑移动平均
#
# 算法
# Y = (X*2 + Y'*(N-1)) / (N+1)
#
# 或者 EMA(X, N) = SMA(X, N+1, 2)
def EMA(vals, n):
    return SMA(vals, n+1, 2)

    '''
    ret = vals[0]
    for x in vals:
        ret = (x * 2 + ret * (n - 1)) / (n + 1)
    return ret
    '''
