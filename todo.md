# 开发笔记

把数据应当分为前市+后市

## 测试模型 回归
通过阅读一天盘式，判断第二天 最高点和收盘涨幅

## 股票筛选模型， 回归
通过昨天后市 + 后市的数据来评估每只股票的1日的获利能力
获利能力为当日晚盘最低点 到次日早盘最高点的差值
尝试输出值为 【最低，最高，价值】
最低，最高 是相对于前市收盘价
价值是最低和最高价的比差

## 买入点判断模型
目标：在晚盘寻找最低点可能会出现的时间

## 卖出点判断模型
目标：在次日前市寻找最高点可能会出现的时间

# 要做的工作

### 调整样本的分布结构
分三组 中小板  沪深300  创业板，采样是这些股票一年的5分钟数据

### 增加指标 来丰富维度
目前来看这样可以增加数据的收敛效果，特别是在精度比较高的时候

### 原始数据导入器
这个工具也要注意排重
可以通过联合UNIQUE主键来解决

### 特征转换工具
这个工具要注意排重
不过可以通过联合UNIQUE主键来解决

### 模型训练器 
每个模型相互分开

# 思考问题

## 关于动态选择缩放参数问题
如果时间跨度太大，那么价格或者交易量都会存在10倍以上的大小变化
所以从每个以月为时间点来提取 最高价格，最低价格，
交易量等信息作为缩放参数，价格浮动每月通常不会有太大变化 差不多10%左右吧
交易量信息变化浮动仍然很大，也许应该使用平均值作为特征缩放因子

## 训练数据的转换太消耗性能了 要想办法优化一下 
流程方面应该优化，应该有个环节从原始数据中把feature提取出来 存在
feature_extracted_stock_trading_5min

## 获取数据有时候崩溃的问题

## 如何续接数据的问题
带有MA的直接做一些提前量就OK
注意哪些EMA的指标，比如MACD的处理
两重循环，外围循环按日期循环，内部循环按股票
这样可以做到运算可分

# 安装部署注意事项

sudo apt-get install libmysqlclient-dev
sudo -H pip3 install mysqlclient


dataset?

Trainings
    fetch_data(type)
    fetch_result(type)
    
# Features reference
0 - open_vec
1 - high_vec
2 - low_vec
3 - close_vec
4 - open_change
5 - high_change
6 - low_change
7 - close_change
8 - ma5
9 - ma15
10 - ma25
11 - ma40
12 - ema_5
13 - ema_15
14 - ema_25
15 - ema_40
16 - boll_up
17 - boll_md
18 - boll_dn
19 - turnover
20 - count
21 - vol
22 - vr
23 - v_ma5
24 - v_ma15
25 - v_ma25
26 - v_ma40
27 - cci_5
28 - cci_15
29 - cci_30
30 - rsi_6
31 - rsi_12
32 - rsi_24
33 - k9
34 - d9
35 - j9
36 - bias_5
37 - bias_10
38 - bias_30
39 - roc_12
40 - roc_25
41 - change
42 - amplitude
43 - amplitude_maxb
44 - amplitude_maxs
45 - wr_5
46 - wr_10
47 - wr_20
48 - mi_5
49 - mi_10
50 - mi_20
51 - mi_30
52 - oscv
53 - dma_dif
54 - dma_ama
55 - ar
56 - br
57 - pdi
58 - mdi
59 - adx
60 - adxr
61 - asi_5
62 - asi_15
63 - asi_25
64 - asi_40
65 - macd_dif
66 - macd_dea
67 - macd_bar
68 - psy
69 - psy_ma
70 - emv_emv
71 - emv_maemv
72 - wvad
73 - wvad_ma