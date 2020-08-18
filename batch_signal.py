#!usr/bin/env python3
# -*- coding: utf-8 -*-

"""
各种技术指标策略的测试

拓展性：
1. 增加策略
    在Signal中增加新的策略逻辑，输出做空和做多信号
2. 增加策略组合方法
    与增加策略修改方法类似，在Signal中增加新的策略逻辑，输出做空和做多信号
"""


from pandas import DataFrame, Series, concat, ExcelWriter, read_excel, set_option, Timestamp, read_csv
from datetime import datetime, timedelta, date
from time import time
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from mpl_finance import candlestick_ohlc
from matplotlib import ticker as mticker
from collections import defaultdict
from multiprocessing import Process, Lock, Manager
from math import pow
from enum import Enum
from tqdm import tqdm
# from progressbar import *

import pymongo
import talib
import os
import pickle
import numpy as np

from position_management import Asset, import_contracts_info
from back_testing import Revenue, RiskMetrics, Cal, Import
from selecting_strategies import selecting_by_weighting



class Arguments():
    """
    外部参数
    """
    def __init__(self, start_date, end_date, target_index, original_asset,
                 transaction_cost_rate, management_argument, lever_ratio=10, short_period=5, 
                 long_period=20, strategy_type=1, period=20, filter_ratio=1, use_ratio = 1, ):
        """
        外部参数赋值给实例属性
        """
        # self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        # self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        # self.ip_address = ip_address
        self.strategy_type = strategy_type
        self.filter_ratio = filter_ratio
        # self.db_name = db_name
        self.target_index = target_index
        self.period = period
        self.short_period = short_period
        self.long_period = long_period
        self.original_asset = original_asset
        self.transaction_fee = transaction_cost_rate
        # self.transaction_cost_rate_of_buying = transaction_cost_rate_of_buying
        # self.transaction_cost_rate_of_selling = transaction_cost_rate_of_selling
        self.lever_ratio = lever_ratio
        # self.use_ratio = use_ratio
        self.withdraw_ratio_critical = management_argument[0]
        self.cost_ratio_critical = management_argument[1]
        self.begin_withdraw = management_argument[2]
        self.position_grade_mode = tuple([1, 0.5, 0])
        # self.position_grade_mode = tuple([1, 0.5])


class Param():
    """
    策略参数
    """
    def __init__(self, strateiges_name):

        # bbands参数
        self.bbands = [6,10,15,20,25,30,40,50,60,80]
        self.bbands_ma = self.bbands_atr = self.bbands_ma_atr = self.bbands

        # # kama_sar_macd_bbands
        # weight_signal = [[1,1,1,1],[2,1,1,1],[1,0,0,2],[1,1,0,1],[2,1,0,1], ]
        # # weight_signal = [[1,1,1,2]]
        # self.kama_sar_macd_bbands = []
        # for i in weight_signal:
        #     for j in self.bbands:
        #         self.kama_sar_macd_bbands.append([i,j])
        # # self.params['kama_sar_macd_bbands'] = [[weight_4, 10]]

        # jma参数
        self.jma = [
                    '[2/31,2/3-2/32,2,60]',
                    '[1/31,1/31-2/3+2/32,1,60]',
                    '[2/31,2/3-2/32,2,45]',
                    '[1/31,1/31-2/3+2/32,1,45]'
                    ]  

        # # bbands_william_sar_sar_ma参数
        # weight_signal = [[1,0,0,1],[1,0,1,0],[1,1,0,0], ]
        # # weight_signal = [[1,0,1,0]]
        # self.bbands_william_sar_sar_ma = []
        # for i in weight_signal: 
        #     for j in self.bbands_atr:
        #         self.bbands_william_sar_sar_ma.append([i,j])

        # kama参数
        self.kama = []
        for period in [10,15,20,25,30,40,50,60,80]:
            # for long_period in np.arange(period,2*period,5):
            #     for short_period in np.arange(0,period,5):
            #         self.kama.append([period, long_period, short_period])
            self.kama.append(period)
        # self.kama_ma = self.kama

        # sar参数
        self.sar = []
        for maximum in [0.1,0.15,0.2,0.25,0.3,0.35]:
            for acceleration in [0.01,0.015,0.02,0.025,0.03]:
                self.sar.append([maximum, acceleration])
        self.sar_ma = self.sar

        # macd参数
        self.macd = []
        for signalperiod in [6,9,12]:
            for fastperiod in np.arange(6,20,3):
                for slowperiod in np.arange(2*fastperiod,3*fastperiod+6,4):
                    self.macd.append([signalperiod, fastperiod, slowperiod])
        # self.macd_ma = self.macd

        # adosc参数
        self.adosc = []
        for fastperiod in [2,3,4,5,6]:
            for slowperiod in np.arange(2*fastperiod,4*fastperiod+6,2):
                self.adosc.append([fastperiod, slowperiod])
        for fastperiod in [8,10]:
            for slowperiod in np.arange(2*fastperiod,4*fastperiod+6,4):
                self.adosc.append([fastperiod, slowperiod])
        self.adosc_ma = self.adosc

        # cci参数
        self.cci = self.cci_ma = list(np.arange(4,20,2)) + list(np.arange(20,50,4))

        # william
        self.william = self.william_ma = list(np.arange(6,20,2)) + list(np.arange(20,50,4))

        # donchian_channel
        self.donchian_channel = self.donchian_channel_ma = [6,10,15,20,25,30,40,50,60,80]
        self.donchian_channel_atr = self.donchian_channel_ma_atr = self.donchian_channel

        # slow_KD
        self.slowKD = []
        for fastk_period in [5,6,8,10,12,15]:
            for slowk_period in [3,4,5]:
                for slowd_period in [3,4,5]:
                    self.slowKD.append([fastk_period, slowk_period, slowd_period])
        self.slowKD_ma = self.slowKD

        # 记录参数个数
        self.numbers = {}
        count = 0
        for attr in dir(self):
            if "__" not in attr and attr != "numbers":
                if isinstance(getattr(self, attr), list):
                    self.numbers[attr] = len(getattr(self, attr))
                else:
                    self.numbers[attr] = 1
                count += self.numbers[attr]
        #         print(attr,"\t",self.numbers[attr])
        # print("sum =", count)

    @classmethod
    def single_param_iniit(cls, index, param):
        # print("cls.{}".format(index) + " = {}".format(param))
        param[index] = param
        # eval("cls.{}".format(index) + " = {}".format(param))

    @classmethod
    def import_specific_strategies(cls, commodities, strategies_file="strategies-30min.csv"):
        """
        2019/03 挑出的低相关性策略
        """
        df = read_csv(strategies_file)
        contract_continues_name = {
                        "rb":"RB.SHF",
                        "ni":"NI.SHF", 
                        "m":"M.DCE",  
                        "sr":"SR.CZC",  
                        "PTA":"TA.CZC",  
                        "GZ300":"IF.CFE",  # 沪深300股指
                        "GZ500":"IC.CFE",  # 中证500股指
                        "GZ50":"IH.CFE",  # 上证50股指
                        "cffex2":"TS.CFE",  # 国债2年期
                        "cffex5":"TF.CFE",  # 国债5年期
                        "cffex10":"T.CFE",   #国债10年期
                        }
        strategies_module = {}
        for commodity in commodities:
            # print(commodity, contract_continues_name[commodity])
            strategies = df[contract_continues_name[commodity]].dropna()
            strategies_module[commodity] = []
            for content in strategies:
                contents = content.split(":")
                index_name = contents[0]
                if isinstance(eval(contents[1]),list):
                    index_param = [float(x) for x in eval(contents[1])]
                else:
                    index_param = [float(contents[1])]
                args_xyz = [float(x) for x in contents[2].split("_")]
                strategies_module[commodity].append([index_name,args_xyz,index_param])
            # print(DataFrame(strategies_module[commodity]))
            # print()
        return strategies_module


class Signal():
    """
    计算各种策略的交易信号
    """
    def cal_filter(self, kama, filter_ratio=1, filter_sampling_days=20):
        """
        kama反转策略的过滤器
        """
        filters = {}
        for i, Date in enumerate(kama.index):
            if i >= filter_sampling_days:
                sample = kama[kama.index[i - filter_sampling_days] : Date]
                filters[Date] \
                = filter_ratio * sample.rolling(2).apply(lambda x: x[1] - x[0], raw=True).dropna().std()
        return Series(filters)

    def kama_turning_with_filter(self, kama, filter_ratio=1):
        """
        带过滤器的kama反转策略，与Kaufman原书内侧策略相同
        """
        filters = self.cal_filter(kama, filter_ratio)
        signal = {}
        for i, Date in enumerate(filters.index):
            if i >= 3:
                today = Date
                yesterday = filters.index[i - 1]
                one_day_before_yesterday = filters.index[i - 2]
                two_days_before_yesterday = filters.index[i - 3]
                if (kama[today] - kama[yesterday]) > filters[today] \
                    or (kama[today] - kama[one_day_before_yesterday]) > filters[today] \
                    or (kama[today] - kama[two_days_before_yesterday]) > filters[today]:
                    signal[today] = 1
                elif (kama[yesterday] - kama[today]) > filters[today] \
                     or (kama[one_day_before_yesterday] - kama[today]) > filters[today] \
                     or (kama[two_days_before_yesterday] - kama[today]) > filters[today]:
                    signal[today] = -1
                else:
                    # signal[today] = signal[yesterday]
                    signal[Date] = 0
            else:
                signal[Date] = 0
        return Series(signal)

    def kama_price_crossing(self, kama, price):
        """
        kama均线与收盘价穿越策略
        
        kama小于收盘价做多，kama大于收盘价做空
        """
        signal = {Date: 1 if kama[Date] < price[Date] else -1 for Date in kama.index}
        return Series(signal)

    def kama_long_short_period_corssing(self, kama_long, kama_short):
        """
        kama金叉死叉策略

        长周期均线小于短周期均线做多，长周期均线大于短周期均线做空
        """
        signal = {Date: 1 if kama_long[Date] < kama_short[Date] else -1 for Date in kama_long.index}
        return Series(signal)

    def sar_price_crossing(self, sar, price):
        """
        sar与收盘价穿越策略

        sar小于收盘价做多，sar大于收盘价做空
        """
        signal = {Date: 1 if sar[Date] < price[Date] else -1 for Date in sar.index}
        return Series(signal)

    def macd_hist_zero(self, macd_hist):
        """
        macd柱策略

        macd柱大于零做多，macd柱小于零做空
        """
        signal = {Date: 1 if macd_hist[Date] >= 0 else -1 for Date in macd_hist.index}
        return Series(signal)

    def atr(self, atr, critical_value):
        """
        atr 阈值判别
        """
        signal = {Date: 1 if atr[Date] <= critical_value else 0.5 for Date in atr.index}
        return Series(signal)

    def atr_long_short_crossing(self, atr_average_short, atr_average_long):
        signal = {Date: 1 if atr_average_short[Date] <= atr_average_long[Date] else 0 for Date in atr_average_short.index}
        return Series(signal)

    def plus_di_average(self, plus_di_average, critical_value):
        signal = {Date: 1 if plus_di_average[Date] >= critical_value else 0 for Date in plus_di_average.index}
        return Series(signal)

    def bbands_price_crossing(self, upper, lower, close_price):
        """
        布林带策略
        """
        signal = {}
        close_price = dict(close_price)
        upper = dict(upper)
        lower = dict(lower)
        # upper_brand = list(upper.values())
        # lower_brand = list(lower.values())
        # close_price与upper日期范围不一致（upper少了最初几天）
        # 所以需要取交集

        for Date in close_price.keys():
            if Date in upper.keys():
                # print(close_price[Date], upper[Date])
                if close_price[Date] >= upper[Date]:
                    signal[Date] = 1
                elif close_price[Date] <= lower[Date]:
                    signal[Date] = -1
                else:
                    signal[Date] = 0
        return Series(signal)

    def jma(self, jma1, jma2, data1, data2):
        """
        JMA策略
        """
        """
        jma1 短周期JMA
        jma2 长周期JMA
        """
        signal = {}
        for i,Date in enumerate(jma2.keys()):
            if i > 0:
                if data1[i] < jma2[Date] and data1[i-1] > jma2[yesterday]:
                    signal[Date] = 1
                elif data1[i] > jma2[Date] and data1[i-1] < jma2[yesterday]:
                    signal[Date] = -1
                else:
                    signal[Date] = 0
                yesterday = Date
            else:
                yesterday = Date
                signal[Date] = 0
        return Series(signal)

    def multi_signal(self, kama, sar, macd_hist, atr, plus_di_average, price):
        """
        多信号策略

        结合 kama反转、sar收盘价穿越、macd柱 策略
        只要三个信号中有一个为多，则做多，否则做空
        """
        critical_atr = 1.7 * atr.mean()
        critical_plus = 0.85 * plus_di_average.mean()
        # 包括atr
        # print("( KAMA + SAR + MACD ) * ATR * PLUS_DI\n")
        # signal = (self.kama_turning_with_filter(kama) + self.sar_price_crossing(sar, price) \
        #          + self.macd_hist_zero(macd_hist)) \
        #          * self.atr(atr, critical_atr) \
        #          * self.plus_di_average(plus_di_average, critical_plus)
        # # 绘图
        # (self.kama_turning_with_filter(kama) + self.sar_price_crossing(sar, price) \
        #          + self.macd_hist_zero(macd_hist)).plot(linestyle='--')
        # self.atr(atr, critical_atr).plot(linestyle='-.')
        # signal.plot()
        # plt.show()
        # signal[signal >= 1] = 1
        # signal[signal <= -1] = -1
        # 不包括atr和plusdi
        signal_kama = self.kama_turning_with_filter(kama)
        signal_sar = self.sar_price_crossing(sar, price)
        signal_macd = self.macd_hist_zero(macd_hist)
        signal =  (signal_kama + signal_sar + signal_macd).dropna()
        signal[signal > -1] = 1
        signal[signal <= -1] = -1
        return signal, tuple([signal_kama,signal_sar,signal_macd])

    def multi_signal_2(self, kama, sar, macd_hist, upper, lower, weight_signal, price, price_average):
        """
        多信号策略

        结合 kama反转、sar收盘价穿越、macd柱、布林带突破 策略
        """
        signal_kama = self.kama_turning_with_filter(kama)
        signal_sar = self.sar_price_crossing(sar, price)
        signal_macd = self.macd_hist_zero(macd_hist)
        signal_bbands = self.bbands_price_crossing(upper, lower, price) 
        # print(weight_signal)
        signal = weight_signal[0]*signal_kama \
                    + weight_signal[1]*signal_sar \
                    + weight_signal[2]*signal_macd \
                    + weight_signal[3]*signal_bbands
        signal = signal / 2

        signal[signal >= 1] = 1
        signal[signal <= -1] = -1
        
        return signal, signal_kama, signal_sar, signal_macd, signal_bbands

    def ma_dx(self, ma_dx):
        """
        动向平均数指标
        """
        signal \
        = {Date: 1 if ma_dx[Date] >= ma_dx.mean() else -1 for Date in ma_dx.index}
        return Series(signal)

    def adosc(self, adosc):
        """
        佳庆指标
        """
        signal \
        = {Date: 1 if adosc[Date] >= 0 else -1 for Date in adosc.index}
        return Series(signal)

    def william(self, willr):
        """
        William指标
        """
        thresh_low = -80
        thresh_high = -20
        signal = {willr.index[0]:0}
        for i,Date in enumerate(willr.index[1:]):
            if willr[i-1] > thresh_high and willr[i] <= thresh_high:
                signal[willr.index[i]] = -1
            elif willr[i-1] < thresh_low and willr[i] >= thresh_low:
                signal[willr.index[i]] = 1
            # elif willr[i-1] < thresh_high and willr[i] >= thresh_high: # new
            #     signal[willr.index[i]] = 1
            # elif willr[i-1] > thresh_low and willr[i] <= thresh_low: # new
            #     signal[willr.index[i]] = -1
            else:
                if i > 0:
                    signal[willr.index[i]] = signal_yesterday
                else:
                    signal[willr.index[i]] = 0
            signal_yesterday = signal[willr.index[i]]
        # print(Series(signal))
        return Series(signal)

    def rsi(self, rsi):
        """
        rsi
        """
        thresh_low = 20
        thresh_high = 80
        signal = {rsi.index[0]:0}
        for i,Date in enumerate(rsi.index[1:]):
            if rsi[i-1] > thresh_high and rsi[i] <= thresh_high:
                signal[rsi.index[i]] = -1
            elif rsi[i-1] < thresh_low and rsi[i] >= thresh_low:
                signal[rsi.index[i]] = 1
            else:
                if i > 0:
                    signal[rsi.index[i]] = signal_yesterday
                else:
                    signal[rsi.index[i]] = 0
            signal_yesterday = signal[rsi.index[i]]
        # signal_plot = list(signal.values())
        # x = []
        # for i,signal_val in enumerate(signal_plot):
        #     x.append(i)
        # plt.plot(x,signal_plot)
        # plt.show()
        return Series(signal)

    def ma_from_ga_demo(self, short_ema, long_ema):
        """
        来自遗传算法demo的ma
        """
        # print(DataFrame([short_ema,long_ema]).T)
        signal \
        = {Date:1 if short_ema[Date] > long_ema[Date] else -1 for Date in long_ema.index}
        return Series(signal)

    def ma_william_rsi(self, short_ema, long_ema, willr, rsi, weight_signal, weight_update):
        # print(weight_signal)
        # print(weight_update)
        signal_ma = Signal().ma_from_ga_demo(short_ema, long_ema).dropna()
        signal_william = Signal().william(willr).dropna()
        signal_rsi = Signal().rsi(rsi).dropna()
        signals = [signal_ma, signal_william, signal_rsi]
        writer = ExcelWriter("signals.xlsx")
        signals_datafram = DataFrame(signals,index=['ma','william','rsi']).T
        signals_datafram.to_excel(writer)
        writer.save()

        if len(weight_update) == 0:  # 只有一个weight
            weight_signal_update = weight_signal
            signal = Series()
            for Date in signals[0].index:
                signal[Date] = 0
            if sum(weight_signal_update[0:3]) > 0:
                i = 0
                while i < 3:
                    if weight_signal_update[i]*weight_signal_update[i*3+3] > 0:
                        if sum(weight_signal_update[3:3:]) == 1:  # 这是唯一一个
                            signal = signals[i]
                            break
                        elif sum(weight_signal_update[3:3:3*i+1]) == 1:  # 这是第一个，因而后面还有
                            signal = signals[i]
                        else:  # 这不是第一个
                            signal = self.combine_signals(signal, signals[i], \
                                                            [weight_signal_update[3*i-2], \
                                                            weight_signal_update[3*i-1]])
                    i += 1
        else:  # 有多个weight隔时换
            signal = Series()
            weight_signal_update = []
            for Date in (signals[0].index & signals[1].index & signals[2].index):
                signal[Date] = 0
                try:
                    weight_signal_update = list(weight_update.loc[Date])
                    # print(weight_signal_update)
                except:
                    pass
                if len(weight_signal_update) > 0:
                    if sum(weight_signal_update[0:3]) > 0:
                        i = 0
                        while i < 3:
                            if weight_signal_update[i]*weight_signal_update[i*3+3] > 0:
                                if sum(weight_signal_update[3:3:]) == 1:  # 这是唯一一个
                                    signal[Date] = signals[i][Date]
                                    break
                                elif sum(weight_signal_update[3:3:3*i+1]) == 1:  # 这是第一个,因而后面还有
                                    signal[Date] = signals[i][Date]
                                else:  # 这不是第一个
                                    signal[Date] = self.combine_signals(signal[Date], signals[i][Date], \
                                                                [weight_signal_update[3*i+1], \
                                                                weight_signal_update[3*i+2]])
                            i += 1
        # print(signal)
        return signal, signal_ma, signal_william, signal_rsi
    
    def combine_signals(self, signal_old, signal_new, relation):
        if isinstance(signal_old, np.int64):
            signal = 0
            if relation == [0,0]:  # 逻辑与
                signal = min(signal_old,signal_new)
            elif relation == [0,1]:  # 逻辑或
                signal = max(signal_old,signal_new)
            elif relation == [1,0]:  # 逻辑异或
                signal = abs(signal_old-signal_new)
            # else:
            #     print("wrong",relation)
        else:
            # signal是一个Series
            signal = Series()
            for Date in signal_old.index:
                signal[Date] = 0
                if Date in signal_new.index:
                    if relation == [0,0]:  # 逻辑与
                        signal[Date] = min(signal_old[Date],signal_new[Date])
                    elif relation == [0,1]:  # 逻辑或
                        signal[Date] = max(signal_old[Date],signal_new[Date])
                    elif relation == [1,0]:  # 逻辑异或
                        signal[Date] = abs(signal_old[Date]-signal_new[Date])
                    # else:
                    #     print("wrong",relation)
        
        return signal

    def cci(self, cci):
        """
        cci策略
        """
        signal = {}
        for Date in cci.keys():
            if cci[Date] > 30:
                signal[Date] = 1
            elif cci[Date] < -30:
                signal[Date] = -1
            else:
                signal[Date] = 0
        return Series(signal)

    def multi_signal_3(self, upper, lower, willr, sar, sar_ma, weight_signal, price, price_average):
        """
        多信号策略
        bbands_atr + william + sar + sar_ma
        """
        signal_bbands = self.bbands_price_crossing(upper, lower, price_average) 
        signal_william = self.william(willr)
        signal_sar = self.sar_price_crossing(sar, price)
        signal_sar_ma = self.sar_price_crossing(sar_ma, price_average)

        signal = weight_signal[0]*signal_bbands \
                    + weight_signal[1]*signal_william \
                    + weight_signal[2]*signal_sar \
                    + weight_signal[3]*signal_sar_ma

        signal[signal >= 1] = 1
        signal[signal <= -1] =-1

        return signal, signal_bbands, signal_william, signal_sar, signal_sar_ma

    def donchian_channel(self, close_price, ma, upper, lower):
        """
        唐奇安通道策略
        """
        signal = {}
        close_price = dict(close_price)
        upper = dict(upper)
        lower = dict(lower)
        # upper_brand = list(upper.values())
        # lower_brand = list(lower.values())
        # close_price与upper日期范围不一致（upper少了最初几天）
        # 所以需要取交集

        for Date in close_price.keys():
            if Date in upper.keys():
                # print(close_price[Date], upper[Date])
                if close_price[Date] >= upper[Date]:
                    signal[Date] = 1
                elif close_price[Date] <= lower[Date]:
                    signal[Date] = -1
                else:
                    signal[Date] = 0
        return Series(signal)

    def slowKD(self, close_price, slowK, slowD):
        signal = Series()
        for i,Date in enumerate(slowK.index):
            if i > 0:
                yesterday = slowK.index[i-1]
                if slowK[Date] > slowD[Date] and slowK[yesterday] <= slowD[yesterday]:
                    signal[Date] = 1
                elif slowK[Date] < slowD[Date] and slowK[yesterday] >= slowD[yesterday]:
                    signal[Date] = -1
                else:
                    signal[Date] = 0
        return signal


class Plot():
    """
    画图
    """
    @classmethod
    def plot(cls, market_data, strategy_name, asset, direction, asset_low, withdraw_line, cost_line, 
                think_of_withdraw_line, positive_direction, negative_direction, path_out, index=[]):
        """
        画图
        从此先作数据预处理，再调用绘图函数
        """
        open_price_plot = list(market_data['Open'])
        open_price_plot = list(open_price_plot[(len(open_price_plot)-len(direction)):])

        high_price_plot = list(market_data['High'])
        high_price_plot = list(high_price_plot[(len(high_price_plot)-len(direction)):])

        low_price_plot = list(market_data['Low'])
        low_price_plot = list(low_price_plot[(len(low_price_plot)-len(direction)):])

        close_price_plot = list(market_data['Close'])
        close_price_plot = list(close_price_plot[(len(close_price_plot)-len(direction)):])
        
        if len(index)>0:
            index_plot = list(index)
            index_plot = [index[0]]*(len(direction)-len(index_plot))+list(index_plot)
        else:
            index_plot = []

        asset_plot = list(asset)
        asset_low_plot = list(asset_low)
        withdraw_line_plot = list(withdraw_line)
        cost_line_plot = list(cost_line)
        think_of_withdraw_line_plot = list(think_of_withdraw_line)

        daysreshape = market_data.reset_index()
        daysreshape.columns = ['DateTime','Open', 'High', 'Low', 'Close', 'Volume']
        daysreshape['DateTime']=mdates.date2num(daysreshape['DateTime'].astype(date))
        del daysreshape['DateTime']
        temp_list = list()
        for i in range(len(daysreshape)-len(positive_direction)):
            temp_list.append(i)
        daysreshape = daysreshape.drop(temp_list)
        daysreshape = daysreshape.reset_index()
        del daysreshape['index']
        daysreshape.insert(0, 'DateTime', daysreshape.index)
        positive_direction_list = list(positive_direction)
        negative_direction_list = list(negative_direction)
        positive_direction_list.pop()
        negative_direction_list.pop()
        positive_direction_list.insert(0, 0)
        negative_direction_list.insert(0, 0)

        # Plot().plot_asset_everyday(asset, asset_low, withdraw_line, cost_line, think_of_withdraw_line)
        Plot().plot_asset(asset_plot, positive_direction_list, negative_direction_list, \
                            strategy_name, daysreshape, open_price_plot, high_price_plot, \
                            low_price_plot, close_price_plot, asset.index, withdraw_line_plot, \
                            cost_line_plot, think_of_withdraw_line_plot, asset_low_plot, \
                            path_out, index_plot)
        # Plot().plot_signals(direction, positive_direction, negative_direction, directions, daysreshape, flag_strategy)

    @classmethod
    def plot_candlestick(cls, market_data):
        daysreshape = market_data.reset_index()
        # daysreshape = daysreshape.rename(columns={'Date': 'DateTime'})
        daysreshape.columns = ['DateTime','Open', 'High', 'Low','Close']
        daysreshape['DateTime']=mdates.date2num(daysreshape['DateTime'].astype(date))
        fig = plt.figure(figsize=(15,10))
        ax1 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
        candlestick_ohlc(ax1, daysreshape.values, width=.6, colorup='#ff1717', colordown='#53c156')
        ax1.grid()
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
        ax1.tick_params(axis='x')

    @classmethod
    def plot_sar(cls, sar):
        plt.scatter(sar.index, sar, marker='.', color='g', label='sar', s=20)

    @classmethod
    def plot_macd(cls, macd, macd_signal, macd_hist):
        macd.plot()
        macd_signal.plot()
        macd_hist.plot()
        
    @classmethod
    def plot_asset(cls, asset_plot, positive_signal, negative_signal, flag_strategy, daysreshape,
                    open_price_plot, high_price_plot, low_price_plot, close_price_plot, xsticks, 
                    withdraw_line_plot, cost_line_plot, think_of_withdraw_line_plot, asset_low_plot,
                    path_out='/pictures', index_plot=[]):
        """
        绘制行情K线图，净资产，每日持仓
        """
        # 根据数据量设计画布尺寸
        if len(asset_plot) < 800:
            fig = plt.figure(figsize=(45, 10), dpi=100)
            asset_line_width = 1.6
            candlestick_width = .6
        elif len(asset_plot) < 1600:
            fig = plt.figure(figsize=(70, 10), dpi=120)
            asset_line_width = 1.5
            candlestick_width = .5
        elif len(asset_plot) < 2400:
            fig = plt.figure(figsize=(100, 10), dpi=150)
            asset_line_width = 1.2
            candlestick_width = .4
        else:
            fig = plt.figure(figsize=(150, 10), dpi=150)
            asset_line_width = 1
            candlestick_width = .3
        ax1 = fig.add_subplot(111)
        ax1.set_ylabel('asset normalized')
        ax1.plot(daysreshape.index,withdraw_line_plot,'orange',linewidth=.6)
        ax1.plot(daysreshape.index-1,think_of_withdraw_line_plot,'green',linewidth=.6)
        ax1.plot(daysreshape.index-0.5,cost_line_plot,'red',linewidth=.6)
        ax1.plot(daysreshape.index+0.5,asset_low_plot,'brown',linewidth=.6)
        ax1.plot(daysreshape.index+0.5,asset_plot,'royalblue',linewidth=asset_line_width)
        ax1.bar(daysreshape.index,positive_signal,alpha=0.7,width=1,facecolor='yellow')
        ax1.bar(daysreshape.index,negative_signal,alpha=0.2,width=1,facecolor='brown')
        y_min_plot = (np.array(asset_plot)).min()-0.2
        y_max_plot = (np.array(asset_plot)).max()+0.1
        plt.ylim(y_min_plot, y_max_plot)
        plt.grid()
        xsticks = [datetime.strftime(x, "%Y-%m-%d") for x in xsticks]
        xsticks_slice = xsticks[::int(len(xsticks)/10)]
        plt.xticks(np.arange(0,len(xsticks),int(len(xsticks)/10)), xsticks_slice)
        plt.gcf().autofmt_xdate()
        
        ax2 = ax1.twinx()  # this is the important function
        # 画指标
        try:
            ax2.plot(daysreshape.index+0.5,index_plot[:-(len(index_plot)-len(daysreshape))],linewidth=1)  # '.',markersize=1.5
        except:
            print("cannot plot index", len(daysreshape.index), len(index_plot))
        candlestick_data = DataFrame([daysreshape.index,open_price_plot,high_price_plot,low_price_plot,close_price_plot])
        candlestick_data = candlestick_data.T
        candlestick_data.columns = ['DateTime', 'Open', 'High', 'Low', 'Close']
        # 绘制K线图
        candlestick_ohlc(ax2, daysreshape.values, width=candlestick_width, colorup='#ff1717', colordown='#53c156')
        ax2.set_ylabel('close price')
        try:
            plt.savefig(path_out+'{}.png'.format(flag_strategy))
        except IOError as temp:
            print(temp)
        # plt.show()
        plt.close()

        # fig = plt.figure(figsize=(45, 10), dpi=100)
        # plt.plot(daysreshape.index,asset_plot,linewidth=asset_line_width)
        # plt.grid()
        # plt.show()

    @classmethod
    def plot_signals(cls, signal, positive_signal, negative_signal, signals, daysreshape, flag_strategy):
        """
        绘制信号图
        """
        signal_plot = list(signal*5+80)
        x = list()
        height = list()
        for i,j in enumerate(signal_plot):
            x.append(i+1)
        fig = plt.figure(figsize=(45, 10), dpi=100)
        ax1 = fig.add_subplot(111)
        ax1.bar(x,positive_signal,alpha=0.7,width=1,facecolor='yellow')
        ax1.bar(x,negative_signal,alpha=0.2,width=1,facecolor='brown')
        plt.grid()
        # print(len(signal_plot),len(positive_signal))
        for i,subsignal in enumerate(signals):
            height.append(200-(i+4)*20)
            # print(len(signal_plot),len(subsignal))
            subsignal_plot = list(subsignal*5 + height[i])
            if len(signal_plot) < len(subsignal):
                subsignal_plot = list(subsignal_plot)[len(subsignal)-len(signal_plot)-1:-1]
            elif len(signal_plot) > len(subsignal):
                subsignal_plot = list([height[i]])*(len(signal_plot)-len(subsignal_plot))+list(subsignal_plot)
            ax1.plot(x,subsignal_plot)
        # ax1.plot(x,signal_plot)
        y_min_plot = height[-1]-30
        y_max_plot = height[0]+30
        plt.ylim(y_min_plot, y_max_plot)

        ax2 = ax1.twinx()  # this is the important function
        # 绘制K线图
        candlestick_ohlc(ax2, daysreshape.values, width=.6, colorup='#ff1717', colordown='#53c156')
        ax2.set_ylabel('close price')
        plt.savefig('pictures/Signal_{}.png'.format(flag_strategy))
        plt.show()
        plt.close()

    @classmethod
    def plot_asset_everyday(cls, asset, asset_low, withdraw_line, cost_line, think_of_withdraw_line):
        asset_everyday = list()
        low_everyday = list()
        withdraw_everyday = list()
        cost_everyday = list()
        think_of_withdraw_everyday = list()
        x_list = list()
        j = 0
        for i,Date in enumerate(asset.index):
            Date = str(Date)
            Date = datetime.strptime(Date, '%Y-%m-%d %H:%M:%S')
            # 提取出每日15点
            if Date.hour == 15:
                j += 1
                asset_everyday.append(asset[Date])
                low_everyday.append(asset_low[Date])
                withdraw_everyday.append(withdraw_line[Date])
                cost_everyday.append(cost_line[Date])
                think_of_withdraw_everyday.append(think_of_withdraw_line[Date])
                x_list.append(j)
        # print(asset_everyday)
        plt.plot(x_list,asset_everyday,'royalblue')
        # plt.plot(x_list,low_everyday,'brown')
        # plt.plot(x_list,withdraw_everyday,'orange')
        # plt.plot(x_list,cost_everyday,'red')
        # plt.plot(x_list,think_of_withdraw_everyday,'green')
        plt.show()


class Strategies():
    @classmethod
    def kama_strategy(cls, args, market_data, param=[20,20,5], strategy_type=1):
        """
        kama策略

        实现三种kama策略
        strategy_type:
        1 - KAMA 反转结合过滤器
        2 - 价格穿越 KAMA
        3 - KAMA 长短周期穿越
        """
        # print("\nkama_strategy\n")
        if strategy_type not in (1, 2, 3):
            print('Strategy Type Setting Error!')
            raise IndexError
        
        (market_data['Close']/market_data['Close'][0]).plot()
        price_close = market_data['Close']

        # close_price_average = Series(talib.MA(np.array(market_data['Close']), timeperiod = 15),index=market_data['Low'].index).dropna()
        # price_close = close_price_average

        if strategy_type == 1 or strategy_type == 2:
            kama \
            = Series(talib.KAMA(np.array(price_close), timeperiod=param), index=price_close.index) \
              .dropna()
        elif strategy_type == 3:
            kama_long \
            = Series(talib.KAMA(np.array(price_close), timeperiod=param[1]), index=price_close.index) \
              .dropna()
            kama_short \
            = Series(talib.KAMA(np.array(price_close), timeperiod=param[2]), index=price_close.index) \
              .dropna()
        if strategy_type == 1:
            signal = Signal().kama_turning_with_filter(kama, args.filter_ratio)
        elif strategy_type == 2:
            signal = Signal().kama_price_crossing(kama, price_close)
        elif strategy_type == 3:
            signal = Signal().kama_long_short_period_corssing(kama_long, kama_short)
        
        # if strategy_type == 1 or strategy_type == 2:
        #     (kama/kama[0]).plot()
        # elif strategy_type == 3:
        #     (kama_long/kama_long[0]).plot()
        #     (kama_short/kama_short[0]).plot()
        # (price_close/price_close[0]).plot()
        # plt.show()
        if strategy_type == 1 or strategy_type == 2:
            return signal, kama
        elif strategy_type == 3:
            return signal, {"long":kama_long, "short":kama_short}

    @classmethod
    def sar_strategy(cls, args, market_data, param=[0.2,0.02]):
        """
        sar策略

        sar与价格相互穿越策略
        """
        # print("\nsar_strategy\n")
        # close_price_average = Series(talib.MA(np.array(market_data['Close']), timeperiod = 10),index=market_data['Low'].index).dropna()
        # close_price_average.plot()
        # high_price_average = Series(talib.MA(np.array(market_data['High']), timeperiod = 10),index=market_data['Low'].index).dropna()
        # low_price_average = Series(talib.MA(np.array(market_data['Low']), timeperiod = 10),index=market_data['Low'].index).dropna()
        sar = Series(talib.SAR(np.array(market_data['High']), np.array(market_data['Low']), maximum=param[0], acceleration=param[1]),
                     index=market_data['Low'].index) \
              .dropna()
        # sar = Series(talib.SAR(np.array(high_price_average), np.array(low_price_average)),
        #              index=low_price_average.index) \
        #       .dropna()
        
        signal = Signal().sar_price_crossing(sar, market_data['Close'])
        # market_data['Close'].plot()
        # sar.plot()

        # plt.legend(["close price","sar"])
        # (signal*120+3000).plot()
        # plt.show()
        return signal, sar

    @classmethod
    def macd_strategy(cls, args, market_data, param=[9,12,26]):
        """
        macd策略
        """
        # print("\nmacd_strategy\n")
        (market_data['Close']/market_data['Close'][0]).plot()
        price_close = market_data['Close']
        # close_price_average = Series(talib.MA(np.array(market_data['Close']), timeperiod = 10),index=market_data['Low'].index).dropna()
        # price_close = close_price_average

        macd, macd_signal, macd_hist = talib.MACD(np.array(price_close), signalperiod=param[0], fastperiod=param[1], slowperiod=param[2])

        macd = Series(macd, index=price_close.index)
        macd_signal = Series(macd_signal, index=price_close.index)
        macd_hist = Series(macd_hist, index=price_close.index)
        # macd.plot()
        # macd_signal.plot()
        # macd_hist.plot()
        # plt.show()
        signal = Signal().macd_hist_zero(macd.dropna())
        return signal, {"macd":macd, "macd_signal":macd_signal, "macd_hist":macd_hist}

    @classmethod
    def atr_strategy(cls, market_data):
        """
        atr策略  真实波动幅度均值 average true range
        """
        atr \
        = Series(talib.ATR(np.array(market_data['Close']), np.array(market_data['High']), np.array(market_data['Low']), ), \
                    index=market_data.index).dropna()
        critical_value = 0.1*atr.mean()
        signal = Signal().atr(atr, critical_value)
        return signal, atr

    @classmethod
    def cci_strategy(cls, market_data, param=14):
        """
        cci策略
        """
        # print("\ncci_strategy\n")
        cci \
        = Series(talib.CCI(np.array(market_data['High']), np.array(market_data['Low']), np.array(market_data['Close']), timeperiod=param), \
                     index=market_data['Close'].index).dropna()
        # plt.plot(list([cci.mean()*20+70000])*(len(market_data['Close'])-len(cci))+list(cci*20+70000))
        # plt.plot(list(market_data['Close']))
        # plt.show()
        signal = Signal().cci(cci).dropna()
        return signal, cci

    @classmethod
    def cmo_strategy(cls, market_data):
        """
        cmo策略
        """
        cmo \
        = Series(talib.CMO(np.array(market_data['Close']), ), index=market_data['Close'].index).dropna()
        cmo.plot()
        plt.show()

    @classmethod
    def adx_strategy(cls, market_data):
        """
        adx策略
        """
        adx \
        = Series(talib.ADX(np.array(market_data['High']), np.array(market_data['Low']), np.array(market_data['Close']), ), \
                     index=market_data['Close'].index).dropna()
        adx.plot()
        plt.show()

    @classmethod
    def adxr_strategy(cls, market_data):
        """
        adxr策略
        """
        adxr \
        = Series(talib.ADXR(np.array(market_data['High']), np.array(market_data['Low']), np.array(market_data['Close']), ), \
                     index=market_data['Close'].index).dropna()
        adxr.plot()
        plt.show()

    @classmethod
    def apo_strategy(cls, market_data):
        """
        apo策略
        """
        apo \
        = Series(talib.APO(np.array(market_data['Close']), ), index=market_data['Close'].index).dropna()
        apo.plot()
        plt.show()

    @classmethod
    def aroon_strategy(cls, market_data):
        """
        aroon策略
        """
        aroon_down, aroon_up \
        = talib.AROON(np.array(market_data['High']), np.array(market_data['Low']))
        aroon_down = Series(aroon_down, index=market_data.index)
        aroon_up = Series(aroon_up, index=market_data.index)
        aroon_down.plot()
        aroon_up.plot()
        plt.show()

    @classmethod
    def bop_strategy(cls, market_data):
        """
        bop策略
        """
        bop \
        = Series(talib.BOP(np.array(market_data['Open']), np.array(market_data['High']), np.array(market_data['Low']), np.array(market_data['Close']), ), \
                     index=market_data['Close'].index).dropna()
        bop.plot()
        plt.show()

    @classmethod
    def dx_strategy(cls, market_data):
        """
        dx策略
        """
        dx \
        = Series(talib.DX(np.array(market_data['High']), np.array(market_data['Low']), np.array(market_data['Close']), ), \
                     index=market_data['Close'].index).dropna()
        dx.plot()
        plt.show()

    @classmethod
    def mfi_strategy(cls, market_data):
        """
       mfi策略
        """
        mfi \
        = Series(talib.MFI(np.array(market_data['High']), np.array(market_data['Low']), \
                            np.array(market_data['Close']), np.array(market_data['Volume'])), \
                     index=market_data['Close'].index).dropna()
        mfi.plot()
        plt.show()

    @classmethod
    def minus_di_strategy(cls, market_data):
        """
        minus_di策略
        """
        minus_di \
        = Series(talib.MINUS_DI(np.array(market_data['High']), np.array(market_data['Low']), \
                            np.array(market_data['Close'])), \
                     index=market_data['Close'].index).dropna()
        minus_di.plot()
        plt.show()

    @classmethod
    def minus_dm_strategy(cls, market_data):
        """
        minus_dm策略
        """
        minus_dm \
        = Series(talib.MINUS_DM(np.array(market_data['High']), np.array(market_data['Low'])), \
                     index=market_data['Close'].index).dropna()
        minus_dm.plot()
        plt.show()

    @classmethod
    def mom_strategy(cls, market_data):
        """
        mom策略
        """
        mom \
        = Series(talib.MOM(np.array(market_data['Close'])), \
                     index=market_data['Close'].index).dropna()
        mom.plot()
        plt.show()

    @classmethod
    def trange_strategy(cls, market_data):
        """
        trange策略
        """
        trange \
        = Series(talib.TRANGE(np.array(market_data['High']), np.array(market_data['Low']), np.array(market_data['Close'])), \
                     index=market_data['Close'].index).dropna()
        trange.plot()
        plt.show()

    @classmethod
    def plus_di_average_strategy(cls, market_data):
        """
        plus_di策略  更向指示器
        """
        print("\nplus_di_strategy:")
        plus_di \
        = Series(talib.PLUS_DI(np.array(market_data['High']), np.array(market_data['Low']), np.array(market_data['Close'])), \
                     index=market_data['Close'].index).dropna()
        plus_di.plot()
        plt.show()
        plus_di_average \
        = Series(talib.MA(np.array(plus_di), timeperiod=60), \
                     index=plus_di.index).dropna()
        plus_di_average.plot()
        plt.show()

    @classmethod
    def bbands_strategy(cls, market_data, param=50):
        """
        布林带突破策略
        """
        # print("\nbbands_strategy\n")
        close_price = market_data['Close']
        upper, middle, lower = talib.BBANDS(np.array(close_price), timeperiod=param)
        upper = Series(upper,index=close_price.index).dropna()
        middle = Series(middle,index=close_price.index).dropna()
        lower = Series(lower,index=close_price.index).dropna()

        signal = Signal().bbands_price_crossing(upper, lower, close_price)

        # 绘图，使非交易日在图上不出现间隔
        # time = range(len(upper))
        # close_price_plot = list(close_price)
        # close_price_plot = list(close_price_plot[(len(close_price_plot)-len(upper)):-1])
        # close_price_plot.append(close_price_plot[-1])
        # upper_plot = list(upper)
        # lower_plot = list(lower)
        # signal_plot = list(signal*100+3000)
        # plt.plot(time,close_price_plot)
        # plt.plot(time,upper_plot)
        # plt.plot(time,lower_plot)
        # plt.plot(time,signal_plot)
        # plt.grid()
        # plt.legend(['close price', 'upper','lower','signal'])
        # plt.show()
        return signal, {"upper":upper, "middle":middle, "lower":lower}

    @classmethod
    def bbands_atr_strategy(cls, market_data, market_data_average, param=10):
        """
        布林带突破策略
        """
        # print("\nbbands_atr_strategy\n")
        close_price = market_data_average['Close']

        upper, middle, lower = talib.BBANDS(np.array(close_price), timeperiod=param)
        upper = Series(upper,index=close_price.index).dropna()
        middle = Series(middle,index=close_price.index).dropna()
        lower = Series(lower,index=close_price.index).dropna()

        atr \
        = Series(talib.ATR(np.array(market_data['Close']), np.array(market_data['High']), np.array(market_data['Low']), \
                 timeperiod=30), index=market_data['Close'].index).dropna()
        close_price = market_data['Close'][len(market_data_average['Close'])-len(close_price):]
        
        ratio = atr/atr.mean()
        upper = ratio*upper+(1-ratio)*middle
        lower = ratio*lower+(1-ratio)*middle
        
        signal = Signal().bbands_price_crossing(upper, lower, close_price)

        return signal, {"upper":upper, "middle":middle, "lower":lower}

    @classmethod
    def jma_strategy(cls, market_data, param):
        """
        JMA策略
        """
        # print("\njma_strategy\n")
        # a = 2/31
        # b = 2/3-2/32
        # c = 2
        # 短周期
        n_short = 5
        # # 长周期
        # n_long = 60
        a,b,c,n_long = eval(param)
        price = market_data['Close']
        dates = list(market_data.index)
        jma_short, data_short = cls.jma(n_short, dates, price, a, b, c)
        jma_long, data_long = cls.jma(n_long, dates, price, a, b, c)

        signal = Signal().jma(jma_short,jma_long, data_short, data_long)

        return signal, \
                {"jma_short":Series(jma_short, index=market_data.index), \
                 "data_short":Series(data_short, index=market_data.index), \
                 "jma_long":Series(jma_long, index=market_data.index), \
                 "data_long":Series(data_long, index=market_data.index)
                }

    @classmethod
    def jma(cls, n, dates, price, a, b, c):
        E = {}
        mu = {}
        mu[dates[n]] = np.mean(np.asarray(price[0:n]))
        for i,Date in enumerate(dates):
            if i >= n+1:
                price_range = np.array(price[i-n:i])
                price_range_latter = np.array(price_range[1:])
                price_range_former = np.array(price_range[0:-1])
                E[Date] = np.abs(np.array(price_range[-1]-price_range[0])) \
                          / np.sum(np.abs(price_range_latter - price_range_former))
                k = 0.6 * pow(a + b * E[Date], c)
                k = max(0, min(k, 1))
                # print(k, price[Date], mu[dates[i-1]])
                mu[dates[i]] = k * price[Date] + (1-k) * mu[dates[i-1]]
        mu.pop(dates[n])
        data_return = price[n+1:]
        return mu, data_return

    @classmethod
    def ma_dx(cls, market_data):
        """
        动向平均数指标
        """
        # print("\nma_dx")
        dx \
        = Series(talib.DX(np.array(market_data['High']), np.array(market_data['Low']), np.array(market_data['Close'])), \
                     index=market_data['Close'].index).dropna()
        ma_dx \
        = Series(talib.MA(np.array(dx), timeperiod=5),index=dx.index).dropna()
        
        plt.plot(list([ma_dx.mean()*200+80000])*(len(market_data['Close'])-len(ma_dx))+list(ma_dx*200+80000))
        plt.plot(list(market_data['Close']))
        plt.show()
        signal = Signal().ma_dx(ma_dx).dropna()
        return signal, ma_dx

    @classmethod
    def adosc(cls, market_data, param=[3,10]):
        """
        佳庆指标
        """
        # print("\nadosc_strategy\n")
        adosc \
        = Series(talib.ADOSC(np.array(market_data['High']), np.array(market_data['Low']), \
                            np.array(market_data['Close']), np.array(market_data['Volume']), \
                            fastperiod=param[0], slowperiod=param[1]), \
                     index=market_data['Close'].index).dropna()

        # plt.plot(list([adosc.mean()/20+80000])*(len(market_data['Close'])-len(adosc))+list(adosc/20+80000))
        # plt.plot(list(market_data['Close']))
        # plt.show()
        signal = Signal().adosc(adosc).dropna()
        return signal, adosc

    @classmethod
    def william(cls, market_data, param=14):
        """
        威廉指标
        """
        # print("\nWilliam_strategy\n")
        william \
        = Series(talib.WILLR(np.array(market_data['High']), np.array(market_data['Low']), \
                            np.array(market_data['Close']), timeperiod=param), \
                                index=market_data['Close'].index).dropna()
        signal = Signal().william(william).dropna()
        return signal, william

    # @classmethod
    # def rsi(cls, market_data):
    #     """
    #     rsi指标
    #     """
    #     # rsi \
    #     # = Series(talib.RSI(np.array(market_data['Close'])), index=market_data['Close'].index).dropna()
    #     up = [0]; down = [0];
    #     for i,Date in enumerate(Delta_price):
    #         if i > 0:
    #             if Delta_price[i] > Delta_price[i-1]:
    #                 up.append(Delta_price[i]-Delta_price[i-1])
    #                 down.append(0)
    #             elif Delta_price[i] < Delta_price[i-1]:
    #                 down.append(Delta_price[i-1]-Delta_price[i])
    #                 up.append(0)
    #             else:
    #                 up.append(0)
    #                 down.appen(0)
    #     up_ave = Series(talib.EMA(np.array(up),timeperiod=12),index=Delta_price.index).dropna()
    #     down_ave = Series(talib.EMA(np.array(down),timeperiod=12),index=Delta_price.index).dropna()
    #     rsi = 100*up_ave/(up_ave+down_ave)
    #     i = 0
    #     for i,Date in enumerate(Delta_price):
    #         if up[i]+down[i] <= 50:
    #             rsi[Date] = 50

    #     signal = Signal().rsi(rsi).dropna()
    #     return signal, Series(rsi)

    @classmethod
    def ma_from_ga_demo(cls, market_data):
        """
        来自遗传算法demo的ma
        """
        short_ema \
        = Series(talib.EMA(np.array(market_data['Close']), timeperiod=12), index=market_data['Close'].index).dropna()
        long_ema \
        = Series(talib.EMA(np.array(market_data['Close']), timeperiod=26), index=market_data['Close'].index).dropna()
        # plt.plot(short_ema)
        # plt.plot(long_ema)
        signal = Signal().ma_from_ga_demo(short_ema, long_ema).dropna()
        return signal, "ma_from_ga_demo"

    @classmethod
    def ma_william_rsi(cls, market_data, market_data_average, market_data_average2, params, j=0, weight_update=[]):
        """
        多指标组合策略
        """
        param = params.params['ma_william_rsi']
        weight_signal = param[j]
        # print(weight_signal)
        short_ema \
        = Series(market_data['Close'])
        long_ema \
        = Series(market_data_average2['Close'])
        william \
        = Series(talib.WILLR(np.array(market_data['High']), np.array(market_data['Low']), \
                            np.array(market_data['Close'])), index=market_data['Close'].index).dropna()
        
        Delta_price = Series(market_data['Close'].sub(market_data_average['Close'],axis=0)).dropna()
        Delta_price = Delta_price.reset_index()
        Delta_price = Series(list(Delta_price['Close']),index=list(Delta_price['Date']))
        # rsi \
        # = Series(talib.RSI(np.array(Delta_price),timeperiod=12), index=Delta_price.index).dropna()
        up = Series(); down = Series();
        for i,Date in enumerate(Delta_price.keys()):
            if i > 0:
                if Delta_price.iloc[i] > Delta_price.iloc[i-1]:
                    up[Date] = Delta_price[i] - Delta_price[i-1]
                    down[Date] = 0
                elif Delta_price.iloc[i] < Delta_price.iloc[i-1]:
                    down[Date] = Delta_price[i-1] - Delta_price[i]
                    up[Date] = 0
                else:
                    up[Date] = 0
                    down[Date] = 0
            else:
                up[Date] = 0
                down[Date] = 0
        up_ave = Series(talib.EMA(np.array(up),timeperiod=12),index=up.keys()).dropna()
        down_ave = Series(talib.EMA(np.array(down),timeperiod=12),index=down.keys()).dropna()
        rsi = Series(100*up_ave/(up_ave+down_ave)).dropna()
        for Date in rsi.keys():
            if up[Date]+down[Date] <= 50 and up[Date]!=0 and down[Date]!=0:
                rsi[Date] = 50
        # rsi_plot = list(rsi)
        # x = []
        # for i,rsi_val in enumerate(rsi_plot):
        #     x.append(i)
        # plt.plot(x,rsi_plot)
        # plt.show()

        signal, signal_ma, signal_william, signal_rsi \
        = Signal().ma_william_rsi(short_ema, long_ema, william, rsi, weight_signal, weight_update)

        signal = signal.dropna()
        signal_ma = signal_ma.dropna()
        signal_william = signal_william.dropna()
        signal_rsi = signal_rsi.dropna()

        return signal, [signal_ma, signal_william, signal_rsi], "ma_william_rsi"

    @classmethod
    def multi_signal_strategy(cls, args, market_data):
        """
        多信号策略

        结合 kama反转、sar收盘价穿越、macd柱 策略
        只要三个信号中有一个为多，则做多，否则做空
        """
        # print("\nmulti_signal_strategy:")
        kama \
        = Series(talib.KAMA(np.array(market_data['Close']), timeperiod=args.period),
                 index=market_data['Close'].index) \
          .dropna()
        
        sar = Series(talib.SAR(np.array(market_data['High']), np.array(market_data['Low'])),
                     index=market_data['Low'].index) \
              .dropna()
        
        _, _, macd_hist = talib.MACD(np.array(market_data['Close']))
        macd_hist = Series(macd_hist, index=market_data['Close'].index)
        
        atr = Series(talib.ATR(np.array(market_data['High']), np.array(market_data['Low']), np.array(market_data['Close'])), \
                     index=market_data['Close'].index).dropna()
        atr_average_short = Series(talib.MA(np.array(atr), timeperiod=5), index=atr.index)
        atr_average_long = Series(talib.MA(np.array(atr), timeperiod=30), index=atr.index)

        plus_di \
        = Series(talib.PLUS_DI(np.array(market_data['High']), np.array(market_data['Low']), np.array(market_data['Close'])), \
                     index=market_data['Close'].index).dropna()
        plus_di_average \
        = Series(talib.MA(np.array(plus_di), timeperiod=60), \
                     index=plus_di.index).dropna()
        # print('plus_di.mean = {}'.format(plus_di.mean()))
        # print('atr.mean = {}'.format(atr.mean()))

        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax1.plot(list(atr_average_short),'royalblue',linewidth=1)
        # ax1.plot(list(plus_di_average),'orange',linewidth=1)
        # ax1.legend("atr","plus_di")
        # ax2 = ax1.twinx()
        # ax2.plot(list(market_data['Close']),'red',linewidth=1)
        # plt.title("atr and plus_di indicator value")
        # plt.show()

        signal, signals = Signal().multi_signal(kama, sar, macd_hist, atr, plus_di_average, market_data['Close'])
        signal = signal.dropna()
        
        return signal, signals, "kama+sar+macd"

    @classmethod
    def multi_signal_strategy_2(cls, args, market_data, market_data_average, params, j):
        """
        多信号策略

        结合 kama反转、sar收盘价穿越、macd柱、布林带穿越策略
        """
        # print("\nmulti_signal_strategy:")
        param = params.params['kama_sar_macd_bbands']
        weight_signal = param[j][0]
        # print(param[j])
        timeperiod_bbands = param[j][1]
        kama \
        = Series(talib.KAMA(np.array(market_data['Close']), timeperiod=args.period),
                 index=market_data['Close'].index) \
          .dropna()
        
        sar = Series(talib.SAR(np.array(market_data_average['High']), np.array(market_data_average['Low'])),
                     index=market_data_average['Low'].index) \
              .dropna()

        _, _, macd_hist = talib.MACD(np.array(market_data['Close']))
        macd_hist = Series(macd_hist, index=market_data['Close'].index)
        # print(market_data_average)
        # print(timeperiod_bbands)
        upper, middle, lower = talib.BBANDS(np.array(market_data_average['Close']), timeperiod=timeperiod_bbands)
        upper = Series(upper,index=market_data_average['Close'].index).dropna()
        middle = Series(middle,index=market_data_average['Close'].index).dropna()
        lower = Series(lower,index=market_data_average['Close'].index).dropna()

        signal, signal_kama, signal_sar, signal_macd, signal_bbands \
        = Signal().multi_signal_2(kama, sar, macd_hist, upper, lower, weight_signal, \
                                    market_data['Close'], market_data_average['Close'])
        
        signal = signal.dropna()
        signal_kama = signal_kama.dropna()
        signal_sar = signal_sar.dropna()
        signal_macd = signal_macd.dropna()
        signal_bbands = signal_bbands.dropna()

        return signal, [signal_kama, signal_sar, signal_macd, signal_bbands], "multi_signal"

    @classmethod
    def kama_bbands_sar(cls, args, market_data, market_data_average, param):
        """
        多信号策略

        结合 kama反转、sar收盘价穿越、macd柱、布林带穿越策略
        """
        # print("\nmulti_signal_strategy:")
        param = params.params['kama_sar_macd_bbands']
        weight_kama_signal = param[j][0]
        timeperiod_bbands = param[j][1]
        kama \
        = Series(talib.KAMA(np.array(market_data['Close']), timeperiod=args.period),
                 index=market_data['Close'].index) \
          .dropna()
        
        sar = Series(talib.SAR(np.array(market_data_average['High']), np.array(market_data_average['Low'])),
                     index=market_data_average['Low'].index) \
              .dropna()

        
        upper, middle, lower = talib.BBANDS(np.array(market_data_average['Close']), timeperiod=timeperiod_bbands)
        upper = Series(upper,index=market_data_average['Close'].index).dropna()
        middle = Series(middle,index=market_data_average['Close'].index).dropna()
        lower = Series(lower,index=market_data_average['Close'].index).dropna()

        signal, signal_kama, signal_sar, signal_macd, signal_bbands \
        = Signal().multi_signal_2(kama, sar, macd_hist, upper, lower, weight_kama_signal, \
                                    market_data['Close'], market_data_average['Close'])
        
        signal = signal.dropna()
        signal_kama = signal_kama.dropna()
        signal_sar = signal_sar.dropna()
        signal_macd = signal_macd.dropna()
        signal_bbands = signal_bbands.dropna()

        return signal, signal_kama, signal_sar, signal_macd, signal_bbands, "multi_signal2"

    @classmethod
    def bbands_atr_and_sar(cls, market_data, market_data_average, timeperiod_bbands):
        """
        多信号策略
        结合 sar收盘价穿越、布林带穿越策略(atr归一化)
        """
        # print("\nbbands_atr + sar\n")
        sar = Series(talib.SAR(np.array(market_data_average['High']), np.array(market_data_average['Low'])),
                     index=market_data_average['Low'].index).dropna()

        close_price = market_data_average['Close']
        upper, middle, lower = talib.BBANDS(np.array(close_price), timeperiod=timeperiod_bbands)
        upper = Series(upper,index=close_price.index).dropna()
        middle = Series(middle,index=close_price.index).dropna()
        lower = Series(lower,index=close_price.index).dropna()

        atr \
        = Series(talib.ATR(np.array(market_data['Close']), np.array(market_data['High']), np.array(market_data['Low']), \
                 timeperiod=30), index=market_data['Close'].index).dropna()
        # print("atr = {}".format(atr.mean()))
        
        ratio = atr/atr.mean()
        upper = ratio*upper+(1-ratio)*middle
        lower = ratio*lower+(1-ratio)*middle

        signal, signal_sar, signal_bbands, \
        = Signal().bbands_atr_and_sar(sar, upper, lower, market_data['Close'], market_data_average['Close'])

        return signal, signal_sar, signal_bbands, "sar+bbands_{}".format(timeperiod_bbands)

    @classmethod
    def bbands_atr_and_william(cls, market_data, market_data_average, timeperiod_bbands):
        """
        组合策略
        布林带突破与william指标策略
        """
        # print("\nbbands_atr + william\n")
        william \
        = Series(talib.WILLR(np.array(market_data['High']), np.array(market_data['Low']), \
                            np.array(market_data['Close'])), index=market_data['Close'].index).dropna()
        # plt.plot(list([william.mean()*100+80000])*(len(market_data['Close'])-len(william))+list(william*100+80000))
        # plt.plot(list(market_data['Close']))
        # plt.show()

        close_price = market_data_average['Close']
        upper, middle, lower = talib.BBANDS(np.array(close_price), timeperiod=timeperiod_bbands)
        upper = Series(upper,index=close_price.index).dropna()
        middle = Series(middle,index=close_price.index).dropna()
        lower = Series(lower,index=close_price.index).dropna()

        atr \
        = Series(talib.ATR(np.array(market_data['Close']), np.array(market_data['High']), np.array(market_data['Low']), \
                 timeperiod=30), index=market_data['Close'].index).dropna()
        # print("atr = {}".format(atr.mean()))
        
        ratio = atr/atr.mean()
        upper = ratio*upper+(1-ratio)*middle
        lower = ratio*lower+(1-ratio)*middle

        signal, signal_william, signal_bbands, \
        = Signal().bbands_atr_and_william(50+william, upper, lower, market_data_average['Close'])

        return signal, "bbands_{}_atr+william".format(timeperiod_bbands)

    @classmethod
    def william_and_sar(cls, market_data, market_data_average):
        """
        组合策略
        william指标与sar指标组合
        """
        # print("\nbbands_atr + william\n")
        william \
        = Series(talib.WILLR(np.array(market_data['High']), np.array(market_data['Low']), \
                        np.array(market_data['Close'])), index=market_data['Close'].index).dropna()
        sar = Series(talib.SAR(np.array(market_data['High']), np.array(market_data['Low'])),
                     index=market_data['Low'].index).dropna()
        signal, signal_william, signal_sar = \
        Signal().william_and_sar(william, sar, market_data['Close'])
        return signal, "william+sar"
    
    @classmethod
    def sar_ma_and_sar(cls, market_data, market_data_average):
        """
        组合策略
        sar_ma与sar指标组合
        """
        sar1 = Series(talib.SAR(np.array(market_data_average['High']), np.array(market_data_average['Low'])),
                     index=market_data_average['Low'].index).dropna()
        sar2 = Series(talib.SAR(np.array(market_data['High']), np.array(market_data['Low'])),
                     index=market_data['Low'].index).dropna()
        signal, signal_sar_ma, signal_sar = \
        Signal().sar_ma_and_sar(sar1, sar2, market_data['Close'], market_data_average['Close'])
        return signal, "sar_ma+sar"

    @classmethod
    def multi_signal_strategy_3(cls, args, market_data, market_data_average, param):
        """
        多信号策略

        结合 kama反转、sar收盘价穿越、macd柱、布林带穿越策略
        """
        # print("\nmulti_signal_strategy:")
        param = params.params['bbands_william_sar_sar_ma']
        weight_signal = param[j][0]
        # print(param[j])
        timeperiod_bbands = param[j][1]

        upper, middle, lower = talib.BBANDS(np.array(market_data_average['Close']), \
                                            timeperiod=timeperiod_bbands)
        upper = Series(upper,index=market_data_average['Close'].index).dropna()
        middle = Series(middle,index=market_data_average['Close'].index).dropna()
        lower = Series(lower,index=market_data_average['Close'].index).dropna()
        atr \
        = Series(talib.ATR(np.array(market_data['Close']), np.array(market_data['High']), np.array(market_data['Low']), \
                 timeperiod=30), index=market_data['Close'].index).dropna()
        
        ratio = atr/atr.mean()
        upper = ratio*upper+(1-ratio)*middle
        lower = ratio*lower+(1-ratio)*middle

        william \
        = Series(talib.WILLR(np.array(market_data['High']), np.array(market_data['Low']), \
                        np.array(market_data['Close'])), index=market_data['Close'].index).dropna()
        
        sar = Series(talib.SAR(np.array(market_data['High']), np.array(market_data['Low'])),
                     index=market_data['Low'].index).dropna()

        sar_ma = Series(talib.SAR(np.array(market_data_average['High']), np.array(market_data_average['Low'])),
                     index=market_data_average['Low'].index).dropna()

        signal, signal_bbands ,signal_william, signal_sar, signal_sar_ma \
        = Signal().multi_signal_3(upper, lower, william, sar, sar_ma, weight_signal, \
                                    market_data['Close'], market_data_average['Close'])
        
        signal = signal.dropna()
        signal_bbands = signal_bbands.dropna()
        signal_william = signal_william.dropna()
        signal_sar = signal_sar.dropna()
        signal_sar_ma = signal_sar_ma.dropna()

        return signal, signal_bbands, signal-william, signal_sar, signal_sar_ma, "multi_signal3"

    @classmethod
    def donchian_channel_strategy(cls, market_data, market_data_average, param=20):
        """
        唐奇安通道策略
        """
        close_price = market_data['Close']
        close_price_ma = market_data_average['Close']
        upper = Series(talib.MAX(np.array(close_price), timeperiod=param), index=close_price.index).dropna()
        lower = Series(talib.MIN(np.array(close_price), timeperiod=param), index=close_price.index).dropna()
        middle = (upper + lower) / 2

        signal = Signal().donchian_channel(close_price, close_price_ma, upper, lower)
        signal = signal.dropna()
        return signal, {"upper":upper, "lower":lower}

    @classmethod
    def donchian_channel_strategy_atr(cls, market_data, market_data_average, param):
        """
        唐奇安通道策略
        """
        close_price = market_data['Close']
        close_price_ma = market_data_average['Close']
        upper = Series(talib.MAX(np.array(close_price), timeperiod=period), index=close_price.index).dropna()
        lower = Series(talib.MIN(np.array(close_price), timeperiod=period), index=close_price.index).dropna()
        middle = (upper + lower) / 2

        atr \
        = Series(talib.ATR(np.array(market_data['Close']), np.array(market_data['High']), np.array(market_data['Low']), \
                 timeperiod=30), index=market_data['Close'].index).dropna()
        close_price = market_data['Close'][len(market_data_average['Close'])-len(close_price):]

        ratio = atr/atr.mean()
        upper = ratio*upper+(1-ratio)*middle
        lower = ratio*lower+(1-ratio)*middle

        signal = Signal().donchian_channel(close_price, close_price_ma, upper, lower)
        signal = signal.dropna()
        return signal, {"upper":donchian_upper, "lower":donchian_lower}
        
    @classmethod
    def slowKD_strategy(cls, market_data, market_data_average, param=[5,3,3]):
        """
        慢速随机指标
        """
        slowK, slowD \
        = talib.STOCH(np.array(market_data['High']), \
                        np.array(market_data['Low']), \
                        np.array(market_data['Close']), \
                        fastk_period=param[0], slowk_period=param[1], slowd_period=param[2])
        K = Series(slowK, index=market_data.index)
        D = Series(slowD, index=market_data.index)
        J = K*3-D*2
        signal = Signal().slowKD(market_data['Close'], K, D)
        return signal, {"K":K, "D":D, "J":J}


class Main():
    """
    主程序
    """
    def __init__(self):
        self.annual_returns = {}
        self.max_drawdowns = {}
        self.vars_ratio = {}
        self.sharpe_ratio = {}
        self.winning_ratio = {}
        self.annual_returns_float = {}
        self.management_arguments = {}
        self.monthly_returns = {}
        self.revenues = DataFrame()
        self.signals = DataFrame()
        self.indexs = DataFrame()

    def test_main(self, if_plot, if_calculate_asset):
        """
        单个策略单个合约单组参数 回测
        """
        param = self.param
        if isinstance(param,list) and len(param) == 1:
            param = param[0]
        args = self.args
        strategy = self.name
        market_data = self.market_data
        market_data_average = self.market_data_average
        market_data_average2 = self.market_data_average2
        trading_dates = self.trading_dates
        index = []
        # 选取策略
        # 若是为了测试，则传入param
        if strategy == 'kama_ma':
            signal, index = Strategies().kama_strategy(args, market_data_average, param)
        elif strategy == 'kama':
            signal, index = Strategies().kama_strategy(args, market_data, param)
        elif strategy == 'sar_ma':
            signal, index = Strategies().sar_strategy(args, market_data_average, param)
        elif strategy == 'sar':
            signal, index = Strategies().sar_strategy(args, market_data, param)
        elif strategy == 'macd_ma':
            signal, index = Strategies().macd_strategy(args, market_data_average, param)
        elif strategy == 'macd':
            signal, index = Strategies().macd_strategy(args, market_data, param)
        elif strategy == 'kama_sar_macd_ma':
            signal, index = Strategies().multi_signal_strategy(args, market_data_average)
        elif strategy == 'kama_sar_macd':
            signal, index = Strategies().multi_signal_strategy(args, market_data)
        elif strategy == 'bbands_ma':
            signal, index = Strategies().bbands_strategy(market_data_average, param)
        elif strategy == 'bbands':
            signal, index = Strategies().bbands_strategy(market_data, param)
        elif strategy == 'jma':
            signal, index = Strategies().jma_strategy(market_data, param)
        elif strategy == 'bbands_ma_atr':
            signal, index \
            = Strategies().bbands_atr_strategy(market_data_average, market_data_average, param)
        elif strategy == 'bbands_atr':
            signal, index \
            = Strategies().bbands_atr_strategy(market_data, market_data_average, param)
        elif strategy == 'adosc':
            signal, index = Strategies().adosc(market_data, param)
        elif strategy == 'adosc_ma':
            signal, index = Strategies().adosc(market_data_average, param)
        elif strategy == 'william':
            signal, index = Strategies().william(market_data, param)
        elif strategy == 'william_ma':
            signal, index, = Strategies().william(market_data_average, param)
        elif strategy == 'cci':
            signal, index = Strategies().cci_strategy(market_data, param)
        elif strategy == 'cci_ma':
            signal, index = Strategies().cci_strategy(market_data_average, param)
        elif strategy == 'rsi':
            signal, index = Strategies().rsi(market_data)
        elif strategy == 'donchian_channel':
            signal, index \
            = Strategies().donchian_channel_strategy(market_data, market_data_average, param)
        elif strategy == 'donchian_channel_ma':
            signal, index \
            = Strategies().donchian_channel_strategy(market_data_average, market_data_average, param)
        elif strategy == 'donchian_channel_atr':
            signal, index \
            = Strategies().donchian_channel_strategy(market_data, market_data_average, param)
        elif strategy == 'donchian_channel_ma_atr':
            signal, index \
            = Strategies().donchian_channel_strategy(market_data_average, market_data_average, param)
        elif strategy == 'slowKD':
            signal, index \
            = Strategies().slowKD_strategy(market_data, market_data_average, param)
        elif strategy == 'slowKD_ma':
            signal, index \
            = Strategies().slowKD_strategy(market_data_average, market_data_average, param)
        else:
            print("strategy={} which is not corrtect !!!".format(strategy))

        if if_calculate_asset == True:
            # 计算每日净资产
            position, asset, direction, positive_direction, negative_direction, withdraw_line, cost_line, \
            think_of_withdraw_line, asset_low, count_abort, count_change, \
            = Asset(args).get_position_and_asset("", signal, market_data, if_plot)
            # , if_change_position_grade=True
            # set_option('display.max_rows', None)
            # print(market_data)

            # 计算绩效指标
            monthly_return, annual_return, max_drawdown, var, sharpe, winning_rate, annual_return_float \
            = Cal.cal_indicators(asset, args.original_asset, signal, market_data)

            # 画图
            if if_plot == True:
                Plot.plot(market_data, strategy_name, asset, direction, asset_low, withdraw_line, \
                        cost_line, think_of_withdraw_line, positive_direction, negative_direction, path_out)
            # 若要计算净值，则不是为了输出信号和指标
            return monthly_return, asset, trading_dates, annual_return, max_drawdown, var, \
                sharpe, winning_rate, annual_return_float

        else:
            # 若不计算净值，则是为了输出信号和指标
            return signal, index

    def test_with_calculate_asset(self, management_argument, if_plot, if_calculate_asset):
        """
        回测中要计算净值绩效
        """
        strategy_name = self.strategy_name
        monthly_return, asset, trading_dates, annual_return, \
        max_drawdown, var, sharpe, winning_rate, annual_return_float \
        = self.test_main(if_plot, if_calculate_asset)
        self.annual_returns[strategy_name] = annual_return
        self.max_drawdowns[strategy_name] = max_drawdown
        self.vars_ratio[strategy_name] = var
        self.sharpe_ratio[strategy_name] = sharpe
        self.winning_ratio[strategy_name] = winning_rate
        self.annual_returns_float[strategy_name] = annual_return_float
        self.management_arguments[strategy_name] = "_".join((str(x) for x in management_argument))
        self.monthly_returns[strategy_name] = monthly_return

    def test_without_calculate_asset(self, if_plot, if_calculate_asset):
        strategy_name = self.strategy_name
        signal, index = self.test_main(if_plot, if_calculate_asset)
        self.signals[strategy_name] = signal # 记录信号
        if not isinstance(index, dict):
            self.indexs[strategy_name] = index
        else:
            for key in list(index.keys()):
                try:
                    self.indexs[strategy_name+":"+key] = index[key]
                except:
                    print(strategy_name+":"+key)
                    print(self.index[key])
                    exit()

    def one_commodity_main(self, code, transcation_fee_at_price, lever_ratio, start_date, end_date, \
                            asset_params, strateiges_name, collection, if_plot, if_calculate_asset, \
                            import_strategies=None, original_asset=1):
        """
        单个合约多组测试
        code是行情数据代码 或 行情数据excel文件名
        """
        # 参数设置
        data_file_folder = "data/"
        data_file = data_file_folder + code + ".xlsx"
        self.strategies = Enum('strategies',strateiges_name)
        # 从excel取行情数据
        # market_data, market_data_average, market_data_average2, trading_dates = Import.import_from_excel(data_file)
        # 从mongoDB取行情数据
        self.market_data, self.market_data_average, self.market_data_average2, self.trading_dates \
        = Import.get_market_data(code, collection, start_date, end_date)
        params = Param(strateiges_name)

        # 设置进度条
        maxval = 0
        for technical_index, _ in self.strategies.__members__.items():
            maxval += params.numbers[technical_index]
        pbar = tqdm(desc=code+"\t", total=len(asset_params)*maxval, leave=False, ncols=100)
        progress = 0
        pbar.update(progress)

        # 多个止损参数组遍历回测
        for i,management_argument in enumerate(asset_params):
            starttime = datetime.now()
            # print(i,management_argument)
            # 设置每个策略的净值计算相关参数
            self.args = Arguments(start_date, end_date, code, original_asset,
                                 transcation_fee_at_price, management_argument, lever_ratio)
            
            # 对指定技术指标的相关策略进行回测
            for self.index,member in self.strategies.__members__.items():
                if params.numbers[self.index] > 1 and if_plot == False:
                    # 对于存在多组参数的策略，遍历每个参数组
                    for param in eval("params.{}".format(self.index)):
                        # 单策略 单合约 单参数组 -> 测试
                        self.name = self.index
                        self.strategy_name = "{}:{}".format(self.index,param)
                        self.param = param
                        if if_calculate_asset == True:
                            self.test_with_calculate_asset(management_argument, if_plot, if_calculate_asset)
                        else:
                            self.test_without_calculate_asset(if_plot, if_calculate_asset)
                        # 更新进度条
                        pbar.update()
                else:
                    # 对于值存在单个参数组的策略 -> 测试
                    if params.numbers[self.index] > 1:
                        param = 14  # 统一在此设置
                    self.name = self.index
                    self.strategy_name = self.index
                    self.param = param
                    if if_calculate_asset == True:
                        self.test_with_calculate_asset(management_argument, if_plot, if_calculate_asset)
                    else:
                        self.test_without_calculate_asset(if_plot, if_calculate_asset)
                    # 更新进度条
                    pbar.update()
                print("  ",code,":",member.value,'=>',self.index)
            
            if if_calculate_asset == True:
                revenue_one_strategy = \
                DataFrame([self.management_arguments, self.annual_returns, self.max_drawdowns, \
                            self.vars_ratio, self.sharpe_ratio, self.winning_ratio, \
                            self.annual_returns_float]).T
                temp = []
                df_index = []
                for i,key in enumerate(self.monthly_returns.keys()):
                    temp.append(self.monthly_returns[key])
                    df_index.append(key)
                revenue_one_strategy.columns \
                = ["argument", "annual_return", "max_drawback", "volatility", "sharpe", "winning_ratio", "annual_return_float"]
                revenue_one_strategy = concat([revenue_one_strategy, DataFrame(temp,index=df_index)], axis=1, sort=False)
                self.revenues = concat([self.revenues,revenue_one_strategy], axis=0, sort=False)
                # self.revenues.drop(self.monthly_returns[key].index[0], axis=1, inplace=True)
            endtime = datetime.now()
            print("\n{}th circle of {} running time {}\n\n".format(i, code, endtime-starttime))
        pbar.close()
        # 若if_calculate_asset==False，则signals, indexs返回空；否则revenues返回空
        return self.revenues, self.signals, self.indexs

    def one_commodity_selected_main(self, code, transcation_fee_at_price, lever_ratio, start_date, end_date, \
                                    collection, if_plot, if_calculate_asset, strategies_dict, original_asset=1):
        """
        从文件导入被选出的策略进行回测
        """
        revenues = DataFrame()
        # 从mongoDB取行情数据
        self.market_data, self.market_data_average, self.market_data_average2, self.trading_dates \
        = Import.get_market_data(code, collection, start_date, end_date)
        for strategy in strategies_dict:
            index, args_xyz, param = strategy
            # param_obj = Param.single_param_iniit(index, param)
            self.name = index
            self.strategy_name = "{}:{}:{}".format(index, param, args_xyz)
            self.param = param
            self.args = Arguments(start_date, end_date, code, original_asset, transcation_fee_at_price, \
                             args_xyz, lever_ratio)
            if if_calculate_asset == True:
                self.test_with_calculate_asset(args_xyz, if_plot, if_calculate_asset)
            else:
                self.test_without_calculate_asset(if_plot, if_calculate_asset)

        if if_calculate_asset == True:
            revenue_one_strategy = \
            DataFrame([self.management_arguments, self.annual_returns, self.max_drawdowns, \
                        self.vars_ratio, self.sharpe_ratio, self.winning_ratio, self.annual_returns_float]).T
            temp = []
            df_index = []
            for i,key in enumerate(self.monthly_returns.keys()):
                temp.append(self.monthly_returns[key])
                df_index.append(key)
            revenue_one_strategy.columns \
            = ["argument", "annual_return", "max_drawback", "volatility", "sharpe", "winning_ratio", "annual_return_float"]
            revenue_one_strategy = concat([revenue_one_strategy, DataFrame(temp,index=df_index)], axis=1, sort=False)
            self.revenues = concat([self.revenues,revenue_one_strategy], axis=0, sort=False)
            # if self.monthly_returns[key].index[0] != "2018-12":
            #     print(key, self.monthly_returns[key].index)
            # self.revenues.drop(self.monthly_returns[key].index[0], axis=1, inplace=True)
        return self.revenues, self.signals, self.indexs


def main(key, commodity, transcation_fees_at_price, lever_ratio, start_date, end_date,
         return_dict, if_plot, if_calculate_asset, strategies_dict, if_import_specific_strategies):
    """
    每个进程跑一个品种
    """
    # 连接数据库
    # db_name = 'CTA_2'
    # ip_address = 'localhost'
    # client = MongoClient('mongodb://{}:27017'.format(ip_address))
    # db = client.get_database(db_name)
    # collection = db.Kline_15_minute

    MG_host = "60.191.32.230"
    MG_port = 27017
    MG_account = "pythonR12Prod"
    MG_pwd = "L2%6*^4"
    dbname = 'His_TICK'
    try:
        client = pymongo.MongoClient(host=MG_host, port=MG_port, connect=False)
    except Exception as e:
        print("mogodb Error")
        print(e)
    db = client[dbname]
    db.authenticate(MG_account, MG_pwd)
    collection = db["Y" + str(start_date.year)]
    print("main task begins")

    if if_import_specific_strategies == False:
        # 批量回测
        # 设置止盈止损参数
        # withdraw_ratio_critical = [0.25]
        # cost_ratio_critical = [0.02,0.03,0.04,0.05]
        # begin_withdraw = [0.02,0.03,0.05,0.07,0.1]
        # asset_params = []
        # for i in withdraw_ratio_critical:
        #     for j in cost_ratio_critical:
        #         for k in begin_withdraw:
        #             asset_params.append([i,j,k])
        asset_params = [[10, 0.99, 10]]
        # print(asset_params)
        # 选取策略指标
        strateiges_name = [
                            'kama',
                            'sar_ma',
                            'sar',
                            'macd',
                            'bbands_ma',  # 9
                            'bbands',  # 10
                            'jma',  # 11
                            'bbands_ma_atr',  # 12
                            'bbands_atr',  # 13
                            'adosc',
                            'adosc_ma',
                            'william',
                            'william_ma',
                            'cci',
                            'cci_ma',
                            # 'rsi',
                            # 'ma_from_ga_demo',
                            # 'ma_william_rsi',
                            # 'kama_sar_macd_bbands',
                            # 'bbands_william_sar_sar_ma',
                            # 'kama_sar_macd_ma',
                            # 'kama_sar_macd',
                            'donchian_channel',
                            'donchian_channel_ma',
                            'donchian_channel_atr',
                            'donchian_channel_ma_atr',
                            'slowKD',
                            'slowKD_ma',
                            ]
        procs = Main()
        revenues_group, signal_group, index_group \
        = procs.one_commodity_main(commodity, transcation_fees_at_price, lever_ratio, start_date, \
                                    end_date, asset_params, strateiges_name, collection, if_plot, \
                                    if_calculate_asset)
    else:
        # 只回测指定策略
        procs = Main()
        revenues_group, signal_group, index_group \
        = procs.one_commodity_selected_main(commodity, transcation_fees_at_price, lever_ratio, \
                                            start_date, end_date, collection, if_plot, \
                                            if_calculate_asset, strategies_dict)
    # 根据绩效作排序
    # sort_strategies, top_strategies = selecting_by_weighting(revenues_group)

    # 收集返回值给共享变量
    return_dict[key] = [revenues_group, signal_group, index_group]


if __name__ == '__main__':
    """
    多合约回测
    """
    # 创建目录
    try:
        os.mkdir("data")
    except:
        pass
    
    # 导入各品种合约信息
    # transcation_fees_at_price, commodity_unit, lever_ratio, commodities = import_contracts_info()

    # 设置回测时间区间
    start_date = datetime(2020, 5, 1, 0, 0)
    end_date = datetime(2020, 7, 31, 15, 00)

    # 是否画图，是否计算净值
    if_plot = False
    if_calculate_asset = True
    if_import_specific_strategies = False

    print("if_plot:",if_plot, "\t", "if_calculate_asset:",if_calculate_asset, "\t", \
            "if_import_specific_strategies:",if_import_specific_strategies)
    if if_calculate_asset == False and if_plot == True:
        print("Wrong !!!")
        exit()

    # 准备并行
    procs = []
    manager = Manager()
    return_dict = manager.dict()

    f = open('all_stocks_1000.pkl','rb')
    code_list = pickle.load(f)
    f.close()
    objects = code_list[:3]
    print(objects)

    # 选取品种
    # objects = ["rb", "ni", "m", "sr", "PTA", "GZ300", "GZ500", "cffex2", "cffex5", "cffex10"]
    # objects = ["rb", "ni", "m", "sr", "PTA", "GZ300", "GZ500", "cffex5", "cffex10"]
    # objects = ["GZ500", "cffex2", "cffex5", "cffex10"]
    # objects = ["ma", "fu", "i"]
    # objects = ["hc", "rm", "j"]
    # objects = ["sm", "c", "jm"]

    if if_import_specific_strategies == True:
        try:
            strategies_imported = Param.import_specific_strategies(objects, "strategies-15min.csv")    # 2019/03 挑出的低相关性策略
            for key in objects:
                print(strategies_imported[key])
        except IOError as temp:
            strategies_imported = {key:[] for key in objects}
            print(temp)
    else:
        strategies_imported = {key:[] for key in objects}
    
    # 每个进程跑一个品种
    for i,key in enumerate(objects):
        p = Process(target=main, 
                    args=\
                        (str(key), str(key), [0,0], 1, start_date, end_date,\
                        return_dict, if_plot, if_calculate_asset, \
                        strategies_imported[key], if_import_specific_strategies))
        print('process {} start at'.format(i+1), datetime.now())
        p.start()
        procs.append(p)
    for i,p in enumerate(procs):
        p.join()
        print('process {} has finished'.format(i+1))
    
    writer_all_strategies = ExcelWriter("data/revenues-{}.xlsx".format(datetime.now().strftime("%Y%m%d-%H%M")))
    signal_all_strategies = ExcelWriter("data/signal-{}.xlsx".format(datetime.now().strftime("%Y%m%d-%H%M")))
    index_all_strategies = ExcelWriter("data/index-{}.xlsx".format(datetime.now().strftime("%Y%m%d-%H%M")))

    # 收集所有进程的返回值
    for key in objects:
        revenues_group, signal_group, index_group = return_dict[key]
        revenues_group.to_excel(writer_all_strategies, sheet_name=str(key))
        signal_group.to_excel(signal_all_strategies, sheet_name=str(key))
        index_group.to_excel(index_all_strategies, sheet_name=str(key))
        
    # 写入excel
    writer_all_strategies.save()
    signal_all_strategies.save()
    index_all_strategies.save()
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
