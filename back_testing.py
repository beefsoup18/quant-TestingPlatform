#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
经营条件选股
"""

import itertools
from pandas import DataFrame, Series, set_option
from pymongo import MongoClient
from datetime import datetime, timedelta
from collections import defaultdict
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
import os
import talib
from sys import version
from xlwt import Workbook

import pandas as pd
import numpy as np
import csv
# import trend_tracking_strategies

from data_module import Importation



class Import():
    
    @classmethod
    def get_market_data(cls, code, collection, start_date=None, end_date=None, content=["Open","High","Low","Close","Volume"], code_str='Stock'):
        """
        获取指定品种的行情数据
        """

        start_time = int(str(start_date.time()).split(".")[0].replace(":","")[:-2])
        end_time = int(str(end_date.time()).split(".")[0].replace(":", "")[:-2])
        print("Symbol:",code, "start_date:",start_date, "end_date:",end_date)
        date = start_date.date()
        date_list = []
        while date <= end_date.date():
            date_list.append(int(str(date).replace("-","")))
            date += timedelta(days=1)

        cond = {code_str: code, "Date":{"$in":date_list}} # "Date":str(start_time.date()).replace("-","")
        show = {"_id": 0, }
        querry = collection.find(cond, show)
        querry_list = []
        querry_list.append(list(querry))
        querry_list = list(itertools.chain.from_iterable(querry_list))
        market_data = pd.DataFrame(querry_list).dropna()
        date_list = list(market_data.Date)
        time_list = list(market_data.Time)

        timestamp = [datetime.strptime(str(date_list[i])+" "+str(time_list[i]), "%Y%m%d %H%M")
                     for i in range(len(time_list))]
        market_data.index = timestamp

        # 滤除没有交易量的数据
        # for timestamp in market_data.index:
        #     if market_data['Volume'][timestamp] == 0:
        #         market_data.drop([timestamp], inplace=True)
        
        # 行情数据取移动平均
        market_data_average = DataFrame()
        timeperiod_ma = 5
        # open_price = Series(talib.EMA(np.array(market_data['Open']), timeperiod=timeperiod_ma),\
        #                 index=market_data['Open'].index).dropna()
        # high_price = Series(talib.EMA(np.array(market_data['High']), timeperiod=timeperiod_ma),\
        #                 index=market_data['High'].index).dropna()
        # low_price = Series(talib.EMA(np.array(market_data['Low']), timeperiod=timeperiod_ma),\
        #                 index=market_data['Low'].index).dropna()
        # close_price = Series(talib.EMA(np.array(market_data['Close']), timeperiod=timeperiod_ma),\
        #                 index=market_data['Close'].index).dropna()
        # market_data_average = DataFrame([open_price, high_price, low_price, close_price])
        # market_data_average = market_data_average.T
        # market_data_average.columns = content

        # MA or EMA
        for key in content:
            market_data_average[key] = \
                Series(talib.SMA(np.array(market_data[key]), timeperiod=timeperiod_ma),\
                        index=market_data.index).dropna()

        timeperiod_ma = 129
        market_data_average2 = DataFrame()
        # open_price = Series(talib.EMA(np.array(market_data['Open']), timeperiod=timeperiod_ma),\
        #                 index=market_data['Open'].index).dropna()
        # high_price = Series(talib.EMA(np.array(market_data['High']), timeperiod=timeperiod_ma),\
        #                 index=market_data['High'].index).dropna()
        # low_price = Series(talib.EMA(np.array(market_data['Low']), timeperiod=timeperiod_ma),\
        #                 index=market_data['Low'].index).dropna()
        # close_price = Series(talib.EMA(np.array(market_data['Close']), timeperiod=timeperiod_ma),\
        #                 index=market_data['Close'].index).dropna()
        # # volume = Series(talib.MA(np.array(market_data['Volume']), timeperiod=timeperiod_ma),\
        # #                 index=market_data['Volume'].index).dropna()
        # market_data_average2 = DataFrame([open_price, high_price, low_price, close_price])
        # market_data_average2 = market_data_average2.T
        # market_data_average2.columns = ["Open","High","Low","Close"]

        trading_dates = market_data.index

        return market_data, market_data_average, market_data_average2, trading_dates

    @classmethod
    def import_from_excel(cls, data_file):
        """
        从excel导入小时数据
        """
        print(data_file)
        df = read_excel(data_file)
        del df['代码']
        del df['名称']
        try:
            del df['持仓量']
        except:
            pass
        try:
            del df['结算价']
        except:
            pass
        try:
            del df['成交额(百万)']
        except:
            pass
        # del df['成交量']
        market_data = df.dropna(axis = 0)
        market_data.columns = ['Date','Open','High','Low','Close','Volume']
        subendtime = datetime.now()
        count_fix = 0
        for i,Date in enumerate(market_data['Date']):
            Date = str(Date)
            Date = datetime.strptime(Date, '%Y-%m-%d %H:%M:%S')
            # # 对于小时线数据：如果出现非整点的时间，则改写为整点
            # if Date.minute > 0:
            #     Date += timedelta(hours = 1)
            #     Date -= timedelta(minutes = Date.minute)
            #     market_data['Date'][i] = Date
            # # 对于30分数据：如果非半小时，则改写为
            # if Date.minute > 0 and Date.minute < 30:
            #     count_fix += 1
            #     Date += timedelta(minutes = (30-Date.minute))
            #     market_data['Date'][i] = Date
            # elif Date.minute > 30:
            #     count_fix += 1
            #     Date += timedelta(hours = 1)
            #     Date -= timedelta(minutes = Date.minute)
            #     market_data['Date'][i] = Date

        market_data.set_index(['Date'], inplace=True)

        # 行情数据取移动平均
        timeperiod_ma = 5
        open_price = Series(talib.EMA(np.array(market_data['Open']), timeperiod=timeperiod_ma),\
                        index=market_data['Open'].index).dropna()
        high_price = Series(talib.EMA(np.array(market_data['High']), timeperiod=timeperiod_ma),\
                        index=market_data['High'].index).dropna()
        low_price = Series(talib.EMA(np.array(market_data['Low']), timeperiod=timeperiod_ma),\
                        index=market_data['Low'].index).dropna()
        close_price = Series(talib.EMA(np.array(market_data['Close']), timeperiod=timeperiod_ma),\
                        index=market_data['Close'].index).dropna()
        # volume = Series(talib.MA(np.array(market_data['Volume']), timeperiod=timeperiod_ma),\
        #                 index=market_data['Volume'].index).dropna()
        # print(len(open_price), len(high_price), len(low_price), len(close_price))
        market_data_average = DataFrame([open_price, high_price, low_price, close_price])
        market_data_average = market_data_average.T
        market_data_average.columns = ["Open","High","Low","Close"]

        timeperiod_ma = 129
        open_price = Series(talib.EMA(np.array(market_data['Open']), timeperiod=timeperiod_ma),\
                        index=market_data['Open'].index).dropna()
        high_price = Series(talib.EMA(np.array(market_data['High']), timeperiod=timeperiod_ma),\
                        index=market_data['High'].index).dropna()
        low_price = Series(talib.EMA(np.array(market_data['Low']), timeperiod=timeperiod_ma),\
                        index=market_data['Low'].index).dropna()
        close_price = Series(talib.EMA(np.array(market_data['Close']), timeperiod=timeperiod_ma),\
                        index=market_data['Close'].index).dropna()
        # volume = Series(talib.MA(np.array(market_data['Volume']), timeperiod=timeperiod_ma),\
        #                 index=market_data['Volume'].index).dropna()
        market_data_average2 = DataFrame([open_price, high_price, low_price, close_price])
        market_data_average2 = market_data_average2.T
        market_data_average2.columns = ["Open","High","Low","Close"]

        trading_dates = market_data.index

        return market_data, market_data_average, market_data_average2, trading_dates


class Cal():
    @classmethod
    def cal_indicators(cls, asset, original_asset, signal, market_data):
        """
        各种绩效计算
        """
        # monthly_return = Revenue.get_monthly_profit_list(asset)
        monthly_return = Revenue.get_monthly_return(asset)
        annual_return = RiskMetrics().get_annual_return(asset)
        monthly_return_std = RiskMetrics().cal_monthly_return_std(monthly_return)
        volatility = np.square(monthly_return_std)
        # volatility = np.square(asset.std())
        max_drawdown = RiskMetrics().cal_max_drawdown(asset)
        sharpe = RiskMetrics().cal_sharpe(monthly_return_std, annual_return)
        # sharpe = RiskMetrics().cal_sharpe(asset.std(), annual_return)
        winning_rate = RiskMetrics().cal_winning_rate(signal, market_data)
        return monthly_return, "{:.2f}%".format(annual_return*100/original_asset), "{:.2f}%".format(max_drawdown*100), \
                "{:.2f}%".format(volatility*100), "{:.3f}".format(sharpe), "{:.2f}%".format(winning_rate*100), \
                annual_return/original_asset
        # return monthly_return, annual_return/original_asset, max_drawdown, \
        #        var, sharpe, winning_rate, annual_return/original_asset


class Revenue():
    """
    收益计算
    """
    @classmethod
    def get_monthly_profit_list(cls, asset):
        """
        根据净值计算每月收益
        """
        monthly_date_list = []  # 换仓日
        Date =  asset.index[0]
        # set_option('display.max_rows', None)
        # 记录每月标签时间戳
        while Date <= asset.index[-1]:
            monthly_return_calculation_date = Date
            # print(monthly_return_calculation_date)
            while monthly_return_calculation_date not in asset.index:
                # and monthly_return_calculation_date < asset.index[-1]:
                # print(monthly_return_calculation_date)
                #  若当天不是交易日,顺延一天
                monthly_return_calculation_date = (monthly_return_calculation_date \
                                                   + timedelta(days=1))
            else:
                monthly_date_list.append(monthly_return_calculation_date)
            Date = Date + relativedelta(months=1)
        # 计算收益
        # print(monthly_date_list)
        monthly_return_dict = {}
        for i, Date in enumerate(monthly_date_list):
            if i == 0:
                monthly_return_dict[Date] = 0
            else:
                # 该日减去上一个月度对应位置日
                monthly_profit = asset[Date] - asset[monthly_date_list[i-1]]
                monthly_return_dict[Date] = monthly_profit / asset[monthly_date_list[i-1]]  # 月度收益率
        return Series(monthly_return_dict)

    @classmethod
    def get_monthly_return(cls, asset):
        """
        根据净值计算月度收益
        """
        year_month = Series([str(x.year)+'0'+str(x.month) if x.month<10 else str(x.year)+str(x.month) for x in asset.index], index=asset.index)
        df = DataFrame([asset, year_month], index=["asset", "year_month"]).T
        sliced_asset = df.groupby(by='year_month')
        last = sliced_asset['asset'].last()
        first = sliced_asset['asset'].first()
        monthly_return = (last - first) / first
        monthly_return.index = [int(x) for x in monthly_return.index]
        monthly_return = monthly_return.sort_index(ascending=True)
        return monthly_return


class Slicer():
    """
    切片器
    """
    def __init__(self, year):
        self.year = year

    def slice(self, timeseries_obj):
        """
        TimeSeries 对象按时间切片
        """
        if isinstance(self.year, int):
            return timeseries_obj[str(self.year)]
        elif self.year == 'all':
            return timeseries_obj
        else:
            raise IndexError   


class RiskMetrics():
    """
    绩效指标计算
    """
    def get_annual_return(self, asset):
        """
        年化收益
        """
        backtesting_days = (asset.index[-1] - asset.index[0]).days + 1
        annual_return = (asset[-1] - asset[0]) / backtesting_days * 365
        # print(asset[0], asset[-1], backtesting_days, annual_return)
        return annual_return

    def cal_monthly_return_std(self, monthly_return):
        """
        月度收益标准差
        """
        return monthly_return.std()

    def cal_max_drawdown(self, asset):
        """
        最大回撤
        """
        n = len(asset)
        g = [0] * n
        max_capital = -1
        for i in range(n - 1):
            max_capital \
            = (max_capital if max_capital > asset[i] else asset[i])
            g[i + 1] = max(g[i], (max_capital - asset[i+1]) / max_capital)
        maximum_drawdown = g[-1]
        return maximum_drawdown

    def cal_sharpe(self, std, annual_return, riskfree_retrun = 0.04):
        """
        夏普比率
        """
        return ((annual_return - riskfree_retrun) / std)
        
    def cal_winning_rate(self, signal, market_data):
        """
        胜率
        """
        # return (len(monthly_return[monthly_return > 0]) / len(monthly_return))
        # asset_diff = Series(np.diff(asset), index=asset.index)
        # return len(asset_diff[asset_diff >= 0]) / (len(asset)-1)
        for i,flag in enumerate(signal):
            if i > 0 and flag == 0:
                signal[i] = signal[i-1]
        close = market_data.Close
        close_diff = Series(np.diff(close), index=market_data.index[:-1])
        close_diff[close_diff>0] = 1
        close_diff[close_diff<0] = -1
        winning = (close_diff - signal).dropna()
        # print(winning)
        # sleep(30)
        return len(winning[winning==0]) / (len(winning)-1)
