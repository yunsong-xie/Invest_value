__author__ = 'Yunsong Xie'
__email__ = 'xiefinance00@gmail.com'
__company__ = 'Xie Finance LLC'

import re, os, sys, datetime, sqlite3
import numpy as np
import pandas as pd
import time
import glob
from matplotlib import pyplot as plt
import lib as common_func

pd.set_option('display.max_column', 60)
pd.set_option('display.max_colwidth', 1200)
pd.set_option('display.width', 12000)


DIR = common_func.misc.get_main_dir()
stock_price = common_func.StockPrice()

dir_fr = f'{DIR}\\static\\Financial_reports\\Wharton\\'
path_fr_db = f'{dir_fr}/fr_wharton.db'


def get_pd_view(pd_data, dict_columns, num=5):
    pd_view = pd_data.iloc[:num].T
    _keys = list(pd_view.keys())
    pd_view = pd_view.reset_index()
    pd_view['code'] = pd_view['index']
    pd_view = pd_view[['code', 'index'] + _keys]
    pd_view['index'] = pd_view['index'].replace(dict_columns)

    pd_view = pd_view.sort_values(by='index')
    pd_view = pd_view.rename(columns={'index': 'col_name'})
    pd_view.index = np.arange(len(pd_view))
    return pd_view


try:
    _ = pd.read_sql("select * from col_name limit 10", con)
except:
    con = sqlite3.connect(path_fr_db)


if __name__ == '__main__0':

    desc_non_null_list = ['Current Assets - Total', 'Current Liabilities - Total',
                          'Cash and Short-Term Investments', 'Operating Activities - Net Cash Flow',
                          'Revenue - Total', 'Cost of Goods Sold', 'Stockholders Equity - Total',
                          'Report Date of Quarterly Earnings']

    desc_output_dict = {'Report Date of Quarterly Earnings': 'rdq',
                        'Current Assets - Total': 'cur_asset',
                        'Current Liabilities - Total': 'cur_liab',
                        'Stockholders Equity - Total': 'book_value',
                        'Cash and Short-Term Investments': 'cash_invest',
                        'Operating Activities - Net Cash Flow': 'cash_flow',
                        'Revenue - Total': 'revenue',
                        'profit': 'profit'}
    desc_positive_list = ['Revenue - Total', 'Stockholders Equity - Total']
    desc_greater_list = []

    desc_new_dict = {'profit': ['Revenue - Total', 'Cost of Goods Sold']}

    year_desc_positive_list = ['Operating Activities - Net Cash Flow', 'profit']
    year_desc_grow_dict = {'ALL': {0: ['Operating Activities - Net Cash Flow', 'Revenue - Total', 'Stockholders Equity - Total',
                                       'profit', 'Cash and Short-Term Investments', 'Current Assets - Total']},
                           0: {0.051: ['Revenue - Total'],
                               0.05: ['Stockholders Equity - Total']},
                           1: {0.035: ['Revenue - Total']},}

    desc_list_all = list(set(desc_non_null_list + list(desc_output_dict) + desc_positive_list))
    _desc_greater_list = []
    for i in desc_greater_list:
        _desc_greater_list += i
    for i in desc_new_dict:
        _desc_greater_list += desc_new_dict[i]
    desc_list_all = list(set(desc_list_all + _desc_greater_list))

    command_query = f"""select col_name, desc from col_name where desc in ("{'", "'.join(desc_non_null_list)}")"""
    pd_col_info = pd.read_sql(command_query, con)

    dict_col_name = pd_col_info.set_index('desc')['col_name'].to_dict()
    dict_col_name_reverse = {dict_col_name[i]: i for i in dict_col_name}
    col_non_null_list = [dict_col_name[i] for i in desc_non_null_list]
    col_output_list = [dict_col_name[i] if i in dict_col_name else i for i in desc_output_dict]
    col_positive_list = [dict_col_name[i] for i in desc_positive_list]
    col_greater_list = [[dict_col_name[i[0]], dict_col_name[i[1]]] for i in desc_greater_list]
    year_col_grow_dict = {i: {} for i in year_desc_grow_dict}
    for key_time in year_desc_grow_dict:
        dict_grow = {}
        for key_grow in year_desc_grow_dict[key_time]:
            dict_grow[key_grow] = [desc_output_dict[i] for i in year_desc_grow_dict[key_time][key_grow]]
        year_desc_grow_dict[key_time] = dict_grow
    year_col_positive_list = [desc_output_dict[i] if i in dict_col_name else i for i in year_desc_positive_list]


    symbols = ['AAPL', 'XOM', 'WMT', 'VTRS', 'ADAP', 'BABA', 'AMD', 'TSLA', 'CSBR', 'CDNA', 'EXPI', 'MIME',
               ]
    symbols = []

    query_symbol_filter = ' where '
    if symbols:
        query_symbol_filter += f"""symbol in ("{'", "'.join(symbols)}") \n"""

    if col_non_null_list:
        if '\n' in query_symbol_filter:
            query_symbol_filter += ' and '
        for i_col, col in enumerate(col_non_null_list):
            if i_col == 0:
                query_symbol_filter += f'{col} is not NULL '
            else:
                query_symbol_filter += f'and {col} is not NULL '
        query_symbol_filter += '\n'

    if col_positive_list:
        if '\n' in query_symbol_filter:
            query_symbol_filter += ' and '
        for i_col, col in enumerate(col_positive_list):
            if i_col == 0:
                query_symbol_filter += f'{col} > 0 '
            else:
                query_symbol_filter += f'and {col} > 0 '
        query_symbol_filter += '\n'

    if col_greater_list:
        if '\n' in query_symbol_filter:
            query_symbol_filter += ' and '
        for i_col, col in enumerate(col_greater_list):
            if i_col == 0:
                query_symbol_filter += f' {col[0]} > {col[1]} '
            else:
                query_symbol_filter += f'and {col[0]} > {col[1]} '
        query_symbol_filter += '\n'

    query_growth_filter, count = ' where ', 0
    for _key_time in year_desc_grow_dict:
        if _key_time == 'ALL':
            key_time_list = [0, 1]
        else:
            key_time_list = [_key_time]
        for key_time in key_time_list:
            query_growth_filter += ' \n '
            for key_grow in year_desc_grow_dict[_key_time]:
                for col in year_desc_grow_dict[_key_time][key_grow]:
                    count += 1
                    if count != 1:
                        query_growth_filter += ' and '
                    query_growth_filter += f'{col}_{key_time} >= {col}_{key_time + 1} * {1 + key_grow}'
    for key_time in [0, 1, 2]:
        for col in year_col_positive_list:
            query_growth_filter += f' and {col}_{key_time} > 0'

    col_query = ', '.join([i for i in col_output_list if i in dict_col_name.values()])
    for key in desc_new_dict:
        key1 = dict_col_name[desc_new_dict[key][0]]
        key2 = dict_col_name[desc_new_dict[key][1]]
        col_query += f', {key1} - {key2} as {key}'
    col_output_list_avg = [i for i in col_output_list if i != 'cshoq']
    col_query_avg_0 = 'min(ty0.rdq) as rdq_0, ' + ', '.join([f'avg(ty0.{i}) as {i}_0' for i in col_output_list_avg if i != 'rdq'])
    col_query_avg_1 = 'min(ty1.rdq) as rdq_1, ' + ', '.join([f'avg(ty1.{i}) as {i}_1' for i in col_output_list_avg if i != 'rdq'])
    col_query_avg_2 = 'min(ty2.rdq) as rdq_2, ' + ', '.join([f'avg(ty2.{i}) as {i}_2' for i in col_output_list_avg if i != 'rdq'])
    col_query_avg_3 = 'min(ty3.rdq) as rdq_3, ' + ', '.join([f'avg(ty3.{i}) as {i}_3' for i in col_output_list_avg if i != 'rdq'])

    query_translate = ' rank, symbol, datafqtr, '
    for key_time in [0, 1, 2, 3]:
        for col in col_output_list_avg:
            if col in dict_col_name_reverse:
                query_translate += f'{col}_{key_time} as {desc_output_dict[dict_col_name_reverse[col]]}_{key_time}, '
            else:
                query_translate += f'{col}_{key_time} as {desc_output_dict[col]}_{key_time}, '
        query_translate += '\n'
    query_translate = query_translate[:-3]


    command_query = f"""with filter_1 as (
        select rank() over (order by symbol, rdq, datafqtr) rank, symbol, rdq, datafqtr, {col_query}
        from report 
        {query_symbol_filter}
        order by symbol, rdq, datafqtr
    ), 
    table_4_year as (
        select t1.rank, t1.symbol, t1.rdq, t1.datafqtr, count(t2.rdq) as num 
        from filter_1 t1, filter_1 t2
        where t1.symbol = t2.symbol
        and t1.rdq > t2.rdq
        and julianday(t1.rdq) - julianday(t2.rdq) <= 1510
        group by t1.symbol, t1.rdq, t1.datafqtr
    ), 
    filter_4_year as (
        select rank, symbol, rdq, datafqtr from table_4_year
        where num >= 16
        order by symbol, rdq, datafqtr
    ),
    data2 as (
        select tf.rank, tf.symbol, tf.datafqtr, 
        {col_query_avg_0}, {col_query_avg_1}, {col_query_avg_2}, {col_query_avg_3}
        from filter_4_year tf, filter_1 ty0, filter_1 ty1, filter_1 ty2, filter_1 ty3
        where tf.symbol = ty0.symbol and tf.symbol = ty1.symbol and tf.symbol = ty2.symbol and tf.symbol = ty3.symbol
        and tf.rank - ty0.rank >= 0 and tf.rank - ty1.rank >= 4 and tf.rank - ty2.rank >=  8 and tf.rank - ty3.rank >=  12
        and tf.rank - ty0.rank <= 3 and tf.rank - ty1.rank <= 7 and tf.rank - ty2.rank <= 11 and tf.rank - ty3.rank >=  15
        group by tf.symbol, tf.rdq, tf.datafqtr
        order by tf.symbol, tf.rdq, tf.datafqtr
    ),
    data_translate as (
        select {query_translate} from data2
    )
    
    select * from data_translate
    {query_growth_filter}
    
    """
    #print(command_query)
    pd_data_ori = pd.read_sql(command_query, con)
    pd_data_ori = pd_data_ori[[i for i in pd_data_ori.keys() if i != 'rank']]

    # Add marketcap info
    pd_marketcap_latest = stock_price.get_marketcap_latest()[['symbol', 'adjclose_latest', 'marketcap_latest']]
    pd_marketcap_report = stock_price.get_price_pd_query(pd_data_ori).rename(columns={'time_request': 'rdq'})
    pd_marketcap_report = pd_marketcap_report[['symbol', 'rdq', 'adjclose']]

    pd_marketcap_report = pd_marketcap_report.merge(pd_marketcap_latest, on='symbol', how='inner')
    pd_marketcap_report['marketcap'] = (pd_marketcap_report['adjclose'] / pd_marketcap_report['adjclose_latest'] *
                                        pd_marketcap_report['marketcap_latest'])
    merge_keys = ['symbol', 'rdq']
    pd_data = pd_data_ori.merge(pd_marketcap_report[merge_keys + ['marketcap']], on=merge_keys, how='inner')
    keys_front = ['symbol', 'rdq', 'datafqtr', 'marketcap']
    pd_data = pd_data[keys_front + [i for i in pd_data.columns if i not in keys_front]]
    pd_data = pd_data.sort_values(by='marketcap')
    pd_data_ori = pd_data.copy()


def plot_dist(pd_data):
    fig, ax = plt.subplots(3, 5, figsize=(15, 7.5))
    titles = ['cur_asset', 'cur_liab', 'cash_flow', 'revenue', 'profit']
    ax = fig.axes
    for year in range(3):
        for i, title_key in enumerate(titles):
            title = f'{title_key}_{year}'
            dist = np.log10(pd_data[title] / pd_data['book_value_0'])
            ax[year * len(titles) + i].hist(dist, bins=35)
            ax[year * len(titles) + i].set_title(title)
    fig.tight_layout()


if 1 == 1:

    mc_book_ratio = [0.35, 75]
    marketcap_min = 100

    pd_data = pd_data_ori.copy()

    pd_data = pd_data.loc[(pd_data.marketcap / pd_data.book_value_0 <= mc_book_ratio[1]) &
                          (pd_data.marketcap / pd_data.book_value_0 >= mc_book_ratio[0])]
    pd_data = pd_data.loc[pd_data.marketcap >= marketcap_min]



if 1 == 0:
    pd_data = pd_data_ori.copy()
    key_head_list = ['rank', 'symbol', 'rdq', 'datafqtr', 'quarter']
    cols = key_head_list + sorted([i for i in list(pd_data.columns) if i not in key_head_list])
    pd_data['quarter'] = pd_data['rdq'].str[:4].astype(int) + ((pd_data['rdq'].str[5:7].astype(int) - 1) // 3) / 4

    pd_data = pd_data[cols]

    pd_price = stock_price.get_price_pd_query(pd_data[['symbol', 'rdq']])
    pd_price_latest = stock_price.get_price_latest(list(pd_data['symbol']))

    pd_select = pd_data.loc[(pd_data.quarter >= 2000) & (pd_data.quarter <= 2003)]
    #pd_data['rdq'] = pd.to_datetime(pd_data['rdq'])

    pd_size = pd_data.groupby('quarter').size().rename('num').reset_index()
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax = fig.axes
    ax[0].plot(pd_size.quarter, pd_size.num, '.-')

if 1 == 0:
    #stock_price = StockPrice()
    pd_data = pd_data_ori.copy()
    pd_price = stock_price.get_price_pd_query(pd_data_ori)

    pd_price = pd_price.rename(columns={'time': 'time_price', 'time_request': 'rdq',  'close': 'price'})
    pd_data = pd_data.merge(pd_price, on=['symbol', 'rdq'], how='inner')
    pd_data['market_cap'] = pd_data['shares'] * pd_data['price']

    key_head_list = ['rank', 'symbol', 'rdq', 'datafqtr', 'time_price', 'price', 'market_cap', 'shares']
    cols = key_head_list + sorted([i for i in list(pd_data.columns) if i not in key_head_list])
    pd_data = pd_data[cols]

# con.close()