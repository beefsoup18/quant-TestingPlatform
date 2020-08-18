#!usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据模块

从数据库导入数据，数据预处理
"""


from pandas import DataFrame
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pymongo import MongoClient

import pandas as pd
import numpy as np



class Importation():
    """
    从数据库导入数据
    """
    def __init__(self, args):
        self.args = args

    def get_trading_dates(self):
        """
        获取交易日元组
        """
        dates = []
        Date = self.args.start_date
        while Date <= self.args.end_date:
            trading \
            = (self.args.db.Index_Market_Data.find_one({'Date': Date})['Whether_Trading_Day'])
            if trading:
                dates.append(Date)
            Date = Date + timedelta(1)
        return tuple(dates)

    def get_stocks_close_price(self, stock_selecting_index_composition_weight,
                               start_date, end_date):
        """
        获取股票收盘价
        """
        stocks_close_price = DataFrame()
        for stock in stock_selecting_index_composition_weight.index:
            stocks_close_price \
            = pd.concat([stocks_close_price,
                         self.get_market_data_one_type(stock, start_date, end_date, 'Close')
                        ],
                        axis=1
                       )
        return stocks_close_price

    def close_pymongo(self):
        """
        断开与mongoDB的连接
        """
        # self.args.db.close()
        self.args.client.close()
        self.args.client = None  # 一定要写这句话，不然系统不会回收，只是关闭了，连接存在。

