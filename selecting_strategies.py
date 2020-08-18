
from pandas import Series, DataFrame, concat, read_excel, ExcelWriter, set_option
from collections import defaultdict
from datetime import datetime


def selecting_by_weighting(df):
    """
    采用加权法筛选最优策略
    """
    if "argument" in list(df.columns) and "param" not in list(df.columns):
        # 将argument插入index
        df_index = []
        for i in range(len(df)):
            df_index.append(df.index[i] + ":" + df.iloc[i]['argument'])
        df.index = df_index
        df.drop('argument', axis=1, inplace=True)
    scores = defaultdict(int)
    for weight, index, ascending in zip([0.7, 0.3, 0],["annual_return_float", "max_drawback", "volatility"], [False, True, True]):
        print(index)
        df = df.sort_values(by=index, axis=0, ascending=ascending)
        df = df.reset_index()
        try:
            df.drop("level_0", axis=1, inplace=True)
        except:
            pass
        df = df.reset_index()
        df.set_index(['index'], inplace=True)
        df_columns = list(df.columns)
        # 该绩效指标下的排名
        df_columns[0] = "th_"+index
        df.columns = df_columns
        for key in df.index:
            scores[key] += weight * df["th_"+index][key]
        df.drop("th_"+index, axis=1, inplace=True)
    best_strategies = concat([DataFrame([scores]).T, df], axis=1, sort=False)
    best_strategies = best_strategies.sort_values(by=0, axis=0, ascending=True)
    best_strategies_top = best_strategies.head(10)
    return best_strategies, best_strategies_top


if __name__ == '__main__':
    """
    挑选最优策略
    """
    set_option("max_columns",5)
    writer_best_strategies = ExcelWriter("best-{}.xlsx".format(datetime.now().strftime("%Y%m%d-%H%M")))
    # for key in ["RB.SHF","NI.SHF","M.DCE","SR.CZC","TA.CZC","TS.CFE","TF.CFE","T.CFE","IF.CFE","IC.CFE"]:
    for key in ["MA.CZC", "FU.SHF", "I.DCE", "HC.SHF", "RM.CZC", "J.DCE"]:
    # for key in ["TS.CFE","TF.CFE","T.CFE"]:
        print(key)
        df = read_excel("new-60min.xlsx", sheet_name=key)
        try:
            df.drop(0, axis=1, inplace=True)
        except:
            pass
        best_strategies, best_strategies_top = selecting_by_weighting(df)
        best_strategies.to_excel(writer_best_strategies, sheet_name=key)
    writer_best_strategies.save()