#!usr/bin/env python3
# -*- coding: utf-8 -*-

"""
仓位管理

"""


from pandas import DataFrame, Series, concat, ExcelWriter, read_excel, set_option
from datetime import datetime, timedelta, date


class Asset():
    """
    净值计算相关
    """
    def __init__(self, args):
        self.args = args

    def get_position_and_asset(self, strategy_name, signal, market_data, if_plot, path_out="", \
                                if_atr_fix=False, if_change_position_grade=False):
        """
        根据交易信号计算每日持仓和净值
        """
        """
        signal 持仓信号字典（1做多 -1做空 0平仓/空仓）
        price 收盘价字典
        """
        if if_plot == True:
            file_log = open(path_out+strategy_name+"-log.csv", "w")
            file_log = open(path_out+strategy_name+"-log.csv", "a")
            print('Date,position,position_grade,price,low,high,asset_low,open_asset,maximum_asset,signal,asset', file=file_log)
        price = market_data['Close']
        high = market_data['High']
        low = market_data['Low']
        asset = {}  # 净值是扣除当日换仓成本后的当日净值
        asset[signal.index[0]] = self.args.original_asset
        position = {}  # 这里的仓位是按当日收盘价换仓后的当日持仓
        position[signal.index[0]] = 0
        count_change = 0
        count_abort = 0
        count_drop = 0
        winning_days = 0
        position_grade = 0
        positive_signal = {}
        negative_signal = {}
        withdraw_line = {}
        cost_line = {}
        think_of_withdraw_line = {}
        asset_low = {}
        maximum_asset = asset[signal.index[0]]

        for i, Date in enumerate(signal.index):
            # print(i,signal[Date])
            # print(Date, Series(asset).index[-1])
            if i > 0:
                yesterday = signal.index[i - 1]
                # 计算当日资产最低值
                # print(yesterday, position[yesterday], signal[yesterday], signal[Date])
                if signal[yesterday] >= 1:
                    asset_low[Date] = asset[yesterday] + position[yesterday] * signal[yesterday] * (low[Date] - price[yesterday])
                elif signal[yesterday] <= -1:
                    asset_low[Date] = asset[yesterday] + position[yesterday] * signal[yesterday] * (high[Date] - price[yesterday])
                else:
                    asset_low[Date] = asset[yesterday]
                # 
                if signal[Date] == 0:
                    signal[Date] = signal[yesterday]

                # 当前仓位的历史最大净值
                maximum_asset = max(maximum_asset, asset[yesterday])
                think_of_withdraw_line[Date] = open_asset * (1+self.args.begin_withdraw)
                if maximum_asset > open_asset * (1+self.args.begin_withdraw):
                    # 净资产已需考虑止盈
                    withdraw_line[Date] = open_asset + (maximum_asset - open_asset) \
                                            * (1 - self.args.withdraw_ratio_critical)  # 止盈线
                else:
                    withdraw_line[Date] = open_asset * (1+self.args.begin_withdraw)
                cost_line[Date] = open_asset * (1-self.args.cost_ratio_critical)
                
                # 止损止盈
                if asset_low[Date] < withdraw_line[Date] and asset[yesterday] > withdraw_line[yesterday] and signal[yesterday] != 0:
                    count_abort += 1
                    signal_last = signal[yesterday]  # 记录平仓前的仓位
                    asset[Date], position[Date] \
                    = self.prevent_loss(Date, date_of_opening, withdraw_line[Date])
                    winning_days = 0
                    signal[Date] = 0
                    if if_change_position_grade == True:
                        position_grade = len(self.args.position_grade_mode)-1
                        # position_grade += 1
                elif asset_low[Date] < cost_line[Date] and signal[yesterday] != 0:
                    count_abort += 1
                    signal_last = signal[yesterday]  # 记录平仓前的仓位
                    asset[Date], position[Date] \
                    = self.prevent_loss(Date, date_of_opening, cost_line[Date])
                    winning_days = 0
                    signal[Date] = 0
                    if if_change_position_grade == True:
                        position_grade = len(self.args.position_grade_mode)-1
                        # position_grade += 1

                else:
                    # 换仓大业！
                    # 计算换仓前当日净值
                    asset_before_changing_position \
                    = asset[yesterday] + position[yesterday] * (price[Date] - price[yesterday]) * signal[yesterday]
                    if signal[yesterday] * (price[Date] - price[yesterday]) > 0:
                        # 若今天能盈利
                        winning_days += 1
                    else:
                        # 若今天亏损
                        winning_days = 0

                    # 换仓 vs 不换仓
                    if signal[Date] == signal[yesterday]:
                        # 不换仓，加不加仓？
                        # winning_days > 2 盈3加仓
                        # winning_days > 4 盈5加仓
                        if winning_days > 2 and position_grade > 0:
                            # 若达到连续判胜次数 且现持仓等级不是最高级 ==> 加仓 
                            winning_days = 0
                            asset[Date], position[Date], position_grade \
                            = self.promote_position(i, price[Date], position_grade, \
                                                    asset_before_changing_position, \
                                                    signal[Date], asset[yesterday], position[yesterday])
                        else:
                            asset[Date] = asset_before_changing_position
                            position[Date] = position[yesterday]
                    else:
                        # signal[Date] ！= signal[yesterday]
                        if signal[yesterday] == 0:
                            # 若前一天是空仓，则当天只能开仓或继续空仓，不能反手
                            open_asset, asset[Date], position[Date], maximum_asset, signal[Date], date_of_opening \
                            = self.build_position(i, Date, price[Date], asset[yesterday], \
                                                    signal[Date], signal[yesterday], position_grade)
                            withdraw_line[Date] = open_asset * (1+self.args.begin_withdraw)
                            asset_low[Date] = open_asset
                        else:
                            # 若前一天不是空仓
                            if signal[Date]*signal[yesterday]<0 and (signal[Date] >= 1 or signal[Date] <= -1): 
                                # 反手(先平，再反向开)
                                count_change += 1
                                # 平仓
                                position_asset = price[Date] * position[yesterday] / self.args.lever_ratio
                                asset[Date] = asset_before_changing_position - position_asset \
                                                + self.closing_position(position_asset, Date, date_of_opening)
                                
                                # 反向开仓
                                open_asset, asset[Date], position[Date], maximum_asset, signal[Date], date_of_opening \
                                = self.build_position(i, Date, price[Date], asset[Date], \
                                                    signal[Date], signal[yesterday], position_grade)
                                if maximum_asset > open_asset * (1+self.args.begin_withdraw):
                                    # 净资产已需考虑止盈
                                    withdraw_line[Date] = open_asset + (maximum_asset - open_asset) \
                                            * (1 - self.args.withdraw_ratio_critical)  # 止盈线
                                else:
                                    withdraw_line[Date] = open_asset * (1+self.args.begin_withdraw)
                                # print("换仓, {}, {:.4f}, {:.4f}, {:.4f}" \
                                # .format(i, withdraw_line[Date], think_of_withdraw_line[Date], cost_line[Date]))
                            else:    
                                # signal[Date] = 0.5 or -0.5 or 0
                                # 不动
                                signal[Date] = signal[yesterday]
                                asset[Date] = asset_before_changing_position
                                position[Date] = position[yesterday]
                    if asset[Date] < asset_min:
                        asset_min = asset[Date]
                # print(position[Date],', ',self.args.position_grade_mode[position_grade] ,\
                #         ', ',price[Date],', ',low[Date],', ',high[Date],', ',asset_low[Date],\
                #         ', ',open_asset,', ',maximum_asset,', ',signal[Date],', ',asset[Date])
                if if_plot == True:
                    print(Date,', ',position[Date],', ',self.args.position_grade_mode[position_grade] ,\
                            ', ',price[Date],', ',low[Date],', ',high[Date],', ',asset_low[Date],\
                            ', ',open_asset,', ',maximum_asset,', ',signal[Date],', ',asset[Date], file=file_log)
            
            else:
                # 第一天 开仓或继续空仓，不可能反手
                # print("开仓： {}  {}  {}  {}  asset: {}".format(i, Date, signal[Date], price[Date], asset[Date]))
                if signal[Date] != 0:
                    open_asset, asset[Date], position[Date], maximum_asset, signal[Date], date_of_opening \
                    = self.build_position(i, Date, price[Date], asset[signal.index[0]], \
                                            signal[Date], 0, position_grade)
                else:
                    open_asset = self.args.original_asset
                    maximum_asset = self.args.original_asset
                    position[Date] = 0
                think_of_withdraw_line[Date] = open_asset * (1+self.args.begin_withdraw)
                withdraw_line[Date] = open_asset * (1+self.args.begin_withdraw)  #止盈线
                cost_line[Date] = open_asset * (1-self.args.cost_ratio_critical)  # 止损线
                asset_low[Date] = open_asset
                asset_min = open_asset
        if if_plot == True:
            file_log.close()
        for Date in signal.index:
            # yesterday = signal.index[i - 1]
            # print(i, "  ",Date,"  ",price[Date] - price[yesterday])
            if signal[Date] >= 1:
                positive_signal[Date] = 5000
                negative_signal[Date] = 0
            elif signal[Date] <= -1:
                positive_signal[Date] = 0
                negative_signal[Date] = 5000
            else:
                positive_signal[Date] = 0
                negative_signal[Date] = 0
        # print("止损止盈平仓次数:{}".format(count_abort))
        # print("主动平仓次数：{}".format(count_drop))
        # print("换仓次数:{}".format(count_change))
        # print('asset_min = ',asset_min)
        return Series(position), Series(asset), Series(signal), Series(positive_signal), \
                Series(negative_signal), Series(withdraw_line), Series(cost_line), \
                Series(think_of_withdraw_line), Series(asset_low), \
                count_abort, count_change

    def build_position(self, i, Date, price_today, asset_today, signal_today, signal_yesterday, position_grade):
        """
        开仓（开仓或继续空仓）
        """
        """
        price_today 今日收盘价
        asset_today 换仓前资产（保证金）
        signal_today 当日持仓信号（1做多 -1做空 0平仓/空仓）
        """ 
        fee_ratio = self.args.transaction_fee[0]
        open_asset = self.args.original_asset
        maximum_asset = self.args.original_asset
        position_today = 0
        if -1 < signal_today < 1:
            # 如果今天无信号，则照搬前一个信号（针对布林带策略开仓后平仓的情形）
            signal_today = signal_yesterday
            
        position_asset = asset_today * self.args.position_grade_mode[position_grade]
        asset_today -= self.deducting_transaction_fee(position_asset, fee_ratio)
        position_asset -= self.deducting_transaction_fee(position_asset, fee_ratio)
        # 开仓资产
        open_asset = asset_today
        maximum_asset = asset_today  # 最大资产重新开始记录
        # 持仓标的物数量（若开仓为正值，若空仓为负值，单位为吨）  
        position_today = position_asset / price_today * self.args.lever_ratio
        # print("开仓, {}, {}, {:.6f}".format(i, position_grade, position_today))
        Date = str(Date)
        Date = datetime.strptime(Date, '%Y-%m-%d %H:%M:%S')
        return open_asset, asset_today, position_today, maximum_asset, signal_today, Date.date

    def prevent_loss(self, Date, date_of_opening, leave_asset):
        """
        平仓
        """
        asset = self.closing_position(leave_asset, Date, date_of_opening)
        position_today = 0
        return asset, position_today

    def promote_position(self, i, price_today, position_grade, asset_before_changing_position, 
                            signal_yesterday, asset_yesterday, position_today):
        """
        加仓（升级）
        """
        fee_ratio = self.args.transaction_fee[0]
        position_asset_before_promotion = position_today * price_today * self.args.lever_ratio
        position_grade -= 1
        position_asset_today = asset_before_changing_position * self.args.position_grade_mode[position_grade]
        if signal_yesterday >= 1 or signal_yesterday <= -1:
            asset_today = asset_before_changing_position - self.deducting_transaction_fee(position_asset_today, fee_ratio)
            position_asset_today -= self.deducting_transaction_fee(position_asset_today, fee_ratio)
        else:
            asset_today = asset_yesterday
        position_today = position_asset_today / price_today * self.args.lever_ratio
        # print("加仓, {}, {}, {:.6f}".format(i,position_grade,position_today))
        return asset_today, position_today, position_grade
        
    def closing_position(self, asset, Date, date_of_opening):
        """
        平仓
        """
        if len(self.args.transaction_fee) > 1:
            # 隔日平仓/当日平仓
            fee_ratio_another_day = self.args.transaction_fee[0]
            fee_ratio_in_day = self.args.transaction_fee[1]

            Date = str(Date)
            Date = datetime.strptime(Date, '%Y-%m-%d %H:%M:%S')
            if Date.date == date_of_opening:
                # 日内平仓
                asset -= self.deducting_transaction_fee(asset, fee_ratio_in_day)
            else:
                # 隔日平仓
                asset -= self.deducting_transaction_fee(asset, fee_ratio_another_day)
        else:
            # 平仓日期不影响手续费
            fee_ratio_another_day = self.args.transaction_fee[0]
            asset -= self.deducting_transaction_fee(asset, fee_ratio_another_day)
        return asset

    def deducting_transaction_fee(self, asset, fee_ratio):
        """
        扣除手续费
        """
        fee = asset * self.args.lever_ratio *  fee_ratio
        return fee


def import_contracts_info():
    """
    导入期货品种合约信息
    """
    # 单位交易额的保证金费率
    transcation_fees_at_price = {
                                    "rb":[0.00012],  # 螺纹钢 0.000012每交易额
                                    "ni":[0.0000127],  # 镍 1.2每手每吨=0.0127%%
                                    "m":[0.00000857],  # 豆粕 0.24每手10吨=0.00857%%
                                    "sr":[0.00006],  # 白糖 3.6每手10吨=0.06%%
                                    "PTA":[0.00012],  # PTA 3.6每手5吨=0.12%%
                                    "GZ300":[0.000028,0.000828],  # 沪深300股指期货
                                    "GZ500":[0.000028,0.000828],  # 中证500股指期货
                                    "GZ50":[0.000028,0.000828],  # 上证50股指期货
                                    "cffex2":[0.000005],  # 2年期国债期货 4.6每一百万票面一手=0.005%%
                                    "cffex5":[0.000005],  # 5年期国债期货 4.6每一百万票面一手=0.005%%
                                    "cffex10":[0.000005],  # 10年期国债期货 4.6每一百万票面一手=0.005%%
                                    "ma":[0.00008,0.00024],  # 甲醇 2.4每手10吨每吨=0.08%%
                                    "fu":[0.00006],  # 燃油 0.00006每交易额
                                    "i":[0.000007],  # 铁矿石 0.000007每交易额
                                    "hc":[0.00012],  # 热轧卷板 
                                    "rm":[0.00075],  # 菜粕
                                    "j":[0.000072],  # 焦炭
                                    "jm":[0.000072],  # 焦煤
                                    "c":[0.000012],  # 玉米 0.24每手10吨
                                    "sm":[0.0001, 0.0002],  # 硅锰 3.6每手5吨
                                }

    # 标的物单位
    commodity_unit = {
                                    "rb":10,  # 螺纹钢 0.000012每交易额
                                    "ni":1,  # 镍 1.2每手每吨=0.0127%%
                                    "m":10,  # 豆粕 0.24每手10吨=0.00857%%
                                    "sr":10,  # 白糖 3.6每手10吨=0.06%%
                                    "PTA":5,  # PTA 3.6每手5吨=0.12%%
                                    "GZ300":300,  # 沪深300股指期货
                                    "GZ500":300,  # 中证500股指期货
                                    "GZ50":300,  # 上证50股指期货
                                    "cffex2":10000,  # 2年期国债期货 4.6每一百万票面一手
                                    "cffex5":10000,  # 5年期国债期货 4.6每一百万票面一手
                                    "cffex10":10000,  # 10年期国债期货 4.6每一百万票面一手
                                    "ma":10,  # 甲醇 2.4每手10吨每吨=0.08%%
                                    "fu":10,  # 燃油 0.00006每交易额
                                    "i":100,  # 铁矿石 0.000007每交易额
                                    "hc": 10,  # 菜粕
                                    "rm": 10,  # 热轧卷板 
                                    "j":100,  # 焦炭
                                    "jm":60,  # 焦煤
                                    "c":10,  # 玉米
                                    "sm":5,  # 硅锰
                                }

    # 杠杆比率
    lever_ratio = {
                    "rb":10,  
                    "ni":10, 
                    "m":10,  
                    "sr":10,  
                    "PTA":10,  
                    "GZ300":5,
                    "GZ500":5,
                    "GZ50":5,
                    "cffex2":66,
                    "cffex5":40,  
                    "cffex10":33, 
                    "ma":14,  # 甲醇 7%
                    "fu":12.5,  # 燃油 8%
                    "i":12.5,  # 铁矿石 8%
                    "hc":10,  # 热轧卷板 
                    "rm":16,  # 菜粕
                    "j":12.5,  # 焦炭
                    "jm":20,  # 焦煤
                    "c":20,  # 玉米
                    "sm":20,  # 硅锰
                    }

    # 各品种合约
    commodities = {
                    "rb":"RB.SHF",
                    "ni":"NI.SHF",
                    "sr":"SR.CZC",
                    "PTA":"TA.CZC",
                    "m":"M.DCE",
                    "GZ300":"IF.CFE",  # 沪深300股指
                    "GZ500":"IC.CFE",  # 中证500股指
                    "GZ50":"IH.CFE",  # 上证50股指
                    "cffex2":"TS.CFE",  # 国债2年期
                    "cffex5":"TF.CFE",  # 国债5年期
                    "cffex10":"T.CFE",  # 国债10年期
                    "ma":"MA.CZC",  # 甲醇
                    "fu":"FU.SHF",  # 燃油
                    "i":"I.DCE",  # 铁矿石
                    "hc":"HC.SHF",  # 热轧卷板
                    "rm":"RM.CZC",  # 菜粕
                    "j":"J.DCE",  # 焦炭
                    "jm":"JM.DCE",  # 焦煤
                    "c":"C.DCE",  # 玉米
                    "sm":"SM.CZC",  # 硅锰
                  }

    return transcation_fees_at_price, commodity_unit, lever_ratio, commodities
    