__author__ = 'Yunsong Xie'
__email__ = 'xiefinance00@gmail.com'
__company__ = 'Xie Finance LLC'

import sqlite3
import numpy as np
import pandas as pd
import glob

import lib as common_func

pd.set_option('display.max_column', 60)
pd.set_option('display.max_colwidth', 1200)
pd.set_option('display.width', 12000)


DIR = common_func.misc.get_main_dir()
stock_price = common_func.StockPrice()

dir_fr = rf'{DIR}\static\Financial_reports\Wharton\\'
path_fr_db = f'{dir_fr}/fr_wharton.db'


try:
    _ = pd.read_sql("select * from col_name limit 10", con)
except:
    con = common_func.misc.get_sql_con()

con_local = sqlite3.connect(path_fr_db)


if __name__ == '__main__':
    pd_symbols = pd.read_sql("select distinct symbol from report", con_local)
    symbols = sorted(pd_symbols['symbol'])
    self = stock_price
    print()

if 1 == 0:
    command = """
    select * from price where symbol = ""
    """
    pd_price = pd.read_sql(command, self.con)

    time_min_1 = '2023-01-01'
    time_min_2 = '2023-02-01'

    price_max_time_range = ['2024-06-20', '2024-07-31']
    dict_price_increase = {
        'time_start_1': '2023-12-01',
        'time_start_2': '2023-12-31',
        'increase': 0.3
    }

    command = f"""
    with price1 as (
        select * from price 
        where time >= {time_min_1}
    ),
    t_current as (
        select max(time) as time_max from price1
    ),
    t_prev as (
        select distinct symbol from price1
        where price1.time >= "{time_min_1}"
        and price1.time <= "{time_min_2}"
    ),
    t_price_peak as (
        select symbol, max(adjclose) as adjclose_peak
        from price1
        where time >= "{price_max_time_range[0]}"
        and time <= "{price_max_time_range[1]}"
        group by symbol
    ),
    t_price_start as (
        select symbol, max(adjclose) as adjclose_start
        from price1
        where time >= "{dict_price_increase['time_start_1']}"
        and time <= "{dict_price_increase['time_start_2']}"
        group by symbol
    ),
    t_price_current as (
        select symbol, adjclose as adjclose_current
        from price1 inner join t_current
        on price1.time = t_current.time_max
    ),
    
    t_price_compile as (
        select t_price_peak.symbol, 
        t_price_peak.adjclose_peak,
        t_price_start.adjclose_start,
        t_price_current.adjclose_current
        from t_price_start inner join t_price_peak
        inner join t_price_current
        inner join t_prev
        on t_price_start.symbol = t_price_peak.symbol
        and t_price_start.symbol = t_price_current.symbol
        and t_price_start.symbol = t_prev.symbol
    )
    select * from t_price_compile
    """
    pd_result_ori = pd.read_sql(command, con_local)


    command_shares = """
    with t_max as (
        select symbol, max(rdq) as max_rqd
        from report
        group by symbol
    )
    select report.symbol, report.CSHOQ as shares
    from report inner join t_max
    where report.symbol = t_max.symbol
    and report.rdq = t_max.max_rqd
    """
    pd_shares = pd.read_sql(command_shares, con_local)

    pd_result = pd_result_ori.merge(pd_shares, on='symbol', how='inner')
    # market in unit of $B
    pd_result['marketcap'] = pd_result['adjclose_peak'] * pd_result['shares'] / 1000
    marketcap_min = 10
    pd_result_1 = pd_result.loc[pd_result['marketcap'] >= marketcap_min]
    pd_result_1 = pd_result_1.sort_values(by='marketcap')

    pd_result_1['ratio_preak_to_start'] = pd_result_1['adjclose_peak'] / pd_result_1['adjclose_start'] - 1
    pd_result_1['ratio_preak_to_current'] = pd_result_1['adjclose_current'] / pd_result_1['adjclose_peak'] - 1

    # pd_result_2 = pd_result_1.loc[pd_result_1['ratio_preak_to_current'] <= -0.1]

    pd_result_2 = pd_result_1.loc[(pd_result_1['ratio_preak_to_current'] <= -0.15) & (pd_result_1['ratio_preak_to_current'] >= -0.22)]
    pd_result_3 = pd_result_1.loc[(pd_result_1['ratio_preak_to_current'] <= -0.2)]
    # pd_result_2 = pd_result_2.loc[pd_result_2['ratio_preak_to_start'] >= 0.4]

    pd_result_2.sort_values(by='ratio_preak_to_current')
    pd_result_2.to_csv(r'C:\Users\yunso\Downloads/short_list.csv', index=False)