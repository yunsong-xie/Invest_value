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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost, scipy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


pd.set_option('display.max_column', 60)
pd.set_option('display.max_colwidth', 2400)
pd.set_option('display.width', 25000)

DIR = common_func.misc.get_main_dir()
stock_price = common_func.StockPrice()

dir_fr = f'{DIR}\\static\\Financial_reports\\Wharton\\'
path_fr_db = f'{dir_fr}/fr_wharton.db'

try:
    _ = pd.read_sql("select * from col_name limit 10", con)
except:
    con = sqlite3.connect(path_fr_db)

query = """ select symbol, rdq from report where actq is NULL """
pd_view = pd.read_sql(query, con)
pd_view = pd_view.sort_values(by=['symbol', 'rdq'])

pd_test = pd.read_csv(r'C:\Users\yunso\Desktop\sbeml7ciogsscc16.csv')
pd_test.datadate.unique()

if __name__ == '__main__':

    n_year_x = 3

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

    year_desc_positive_list = ['Operating Activities - Net Cash Flow', 'profit', 'Cash and Short-Term Investments']
    year_desc_grow_dict = {'ALL': {0: ['Operating Activities - Net Cash Flow', 'Revenue - Total', 'Stockholders Equity - Total',
                                       'profit', 'Cash and Short-Term Investments', 'Current Assets - Total']},
                           0: {0.001: ['Revenue - Total'],
                               0.0015: ['Stockholders Equity - Total']},
                           1: {0: ['Revenue - Total']}, }

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
    dict_col_name['profit'] = 'profit'
    dict_col_name_reverse = {dict_col_name[i]: i for i in dict_col_name}
    col_non_null_list = [dict_col_name[i] for i in desc_non_null_list]
    col_output_list = [dict_col_name[i] if i in dict_col_name else i for i in desc_output_dict]
    col_positive_list = [dict_col_name[i] for i in desc_positive_list]
    col_greater_list = [[dict_col_name[i[0]], dict_col_name[i[1]]] for i in desc_greater_list]
    year_col_grow_dict = {i: {} for i in year_desc_grow_dict}
    year_desc_grow_dict_sql = {}
    for key_time in year_desc_grow_dict:
        dict_grow = {}
        for key_grow in year_desc_grow_dict[key_time]:
            dict_grow[key_grow] = [dict_col_name[i] for i in year_desc_grow_dict[key_time][key_grow]]
        year_desc_grow_dict_sql[key_time] = dict_grow
    year_col_positive_list = [dict_col_name[i] if i in dict_col_name else i for i in year_desc_positive_list]

    symbols = ['AAPL', 'XOM', 'WMT', 'WERN', 'STE', 'HLT']
    #symbols = []

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
    for _key_time in year_desc_grow_dict_sql:
        if _key_time == 'ALL':
            key_time_list = [0, 1]
        else:
            key_time_list = [_key_time]
        for key_time in key_time_list:
            query_growth_filter += ' \n '
            for key_grow in year_desc_grow_dict_sql[_key_time]:
                for col in year_desc_grow_dict_sql[_key_time][key_grow]:
                    count += 1
                    if count != 1:
                        query_growth_filter += ' and '
                    query_growth_filter += f'{col}_{key_time} >= {col}_{key_time + 1} * {1 + key_grow}'
    year_list_1 = list(range(n_year_x))
    for key_time in year_list_1:
        for col in year_col_positive_list:
            query_growth_filter += f' and {col}_{key_time} > 0'

    col_query = ', '.join([i for i in col_output_list if (i in dict_col_name.values()) & (i not in ['rdq', 'profit'])])
    for key in desc_new_dict:
        key1 = dict_col_name[desc_new_dict[key][0]]
        key2 = dict_col_name[desc_new_dict[key][1]]
        col_query += f', {key1} - {key2} as {key}'
    col_output_list_avg = [i for i in col_output_list if i != 'cshoq']
    col_query_avg_0 = 'max(ty0.rdq) as rdq_0, ' + ', '.join([f'avg(ty0.{i}) as {i}_0' for i in col_output_list_avg if i != 'rdq'])
    col_query_avg_1 = 'max(ty1.rdq) as rdq_1, ' + ', '.join([f'avg(ty1.{i}) as {i}_1' for i in col_output_list_avg if i != 'rdq'])
    col_query_avg_2 = 'max(ty2.rdq) as rdq_2, ' + ', '.join([f'avg(ty2.{i}) as {i}_2' for i in col_output_list_avg if i != 'rdq'])
    col_query_avg_q0 = 'max(tq.rdq) as rdq_q0, ' + ', '.join([f'avg(tq.{i}) as {i}_q0' for i in col_output_list_avg if i != 'rdq'])
    col_query_avg_q1 = 'max(tq.rdq) as rdq_q1, ' + ', '.join([f'avg(tq.{i}) as {i}_q1' for i in col_output_list_avg if i != 'rdq'])
    col_query_avg_q4 = 'max(tq.rdq) as rdq_q4, ' + ', '.join([f'avg(tq.{i}) as {i}_q4' for i in col_output_list_avg if i != 'rdq'])

    col_query_avg_pq1 = 'max(tq.rdq) as rdq_pq1, ' + ', '.join([f'avg(tq.{i}) as {i}_pq1' for i in col_output_list_avg if i != 'rdq'])
    col_query_avg_pq2 = 'max(tq.rdq) as rdq_pq2, ' + ', '.join([f'avg(tq.{i}) as {i}_pq2' for i in col_output_list_avg if i != 'rdq'])
    col_query_avg_pq3 = 'max(tq.rdq) as rdq_pq3, ' + ', '.join([f'avg(tq.{i}) as {i}_pq3' for i in col_output_list_avg if i != 'rdq'])
    col_query_avg_pq4 = 'max(tq.rdq) as rdq_pq4, ' + ', '.join([f'avg(tq.{i}) as {i}_pq4' for i in col_output_list_avg if i != 'rdq'])


    merge_query_x, merge_query_x_filter = '', ''
    year_list = list(range(n_year_x))
    for key_time in year_list:
        for key_item in col_output_list_avg:
            merge_query_x_filter += f', ty{key_time}.{key_item}_{key_time}'
            merge_query_x += f', ty{key_time}.{key_item}_{key_time}'
    quarter_list = ['q0', 'q1', 'q4']
    for key_time in quarter_list:
        for key_item in col_output_list_avg:
            merge_query_x += f', t{key_time}.{key_item}_{key_time}'

    p_list = ['pq1', 'pq2', 'pq3', 'pq4']
    dict_merge_query_y = {}
    for key_time in p_list:
        merge_query_y = ''
        for key_item in col_output_list_avg:
            merge_query_y += f', ty{key_time}.{key_item}_{key_time}'
        dict_merge_query_y[key_time] = merge_query_y

    query_translate = ' rank, symbol, datafqtr, '
    for key_time in year_list + quarter_list + p_list:
        for col in col_output_list_avg:
            if col in dict_col_name_reverse:
                query_translate += f'{col}_{key_time} as {desc_output_dict[dict_col_name_reverse[col]]}_{key_time}, '
            else:
                query_translate += f'{col}_{key_time} as {desc_output_dict[col]}_{key_time}, '
        query_translate += '\n'
    query_translate = query_translate[:-3]

    command_query = f"""
    with filter_0 as (
        select rank() over (order by symbol, rdq, datafqtr) rank, symbol, rdq, datafqtr, cogsq, {col_query}
        from report
        order by symbol, rdq, datafqtr
    ), 
    filter_1 as (
        select *
        from filter_0 
        {query_symbol_filter}
        order by symbol, rdq, datafqtr
    ),
    table_3_year as (
        select t1.rank, t1.symbol, t1.rdq, t1.datafqtr, count(t2.rdq) as num 
        from filter_1 t1, filter_1 t2
        where t1.symbol = t2.symbol
        and t1.rdq > t2.rdq
        and julianday(t1.rdq) - julianday(t2.rdq) <= 1135
        group by t1.symbol, t1.rdq, t1.datafqtr
    ), 
    filter_3_year as (
        select rank, symbol, rdq, datafqtr from table_3_year
        where num = 12
        order by symbol, rdq, datafqtr
    ),
    data0_filter as (
        select tf.rank, tf.symbol, tf.datafqtr, tf.rdq, {col_query_avg_0}
        from filter_3_year tf, filter_1 ty0
        where tf.symbol = ty0.symbol 
        and tf.rank - ty0.rank >= 0 
        and tf.rank - ty0.rank <= 3 
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    data1_filter as (
        select tf.rank, tf.symbol, tf.rdq, {col_query_avg_1}
        from filter_3_year tf, filter_1 ty1
        where tf.symbol = ty1.symbol
        and tf.rank - ty1.rank >= 4
        and tf.rank - ty1.rank <= 7
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    data2_filter as (
        select tf.rank, tf.symbol, tf.rdq, {col_query_avg_2}
        from filter_3_year tf, filter_1 ty2
        where tf.symbol = ty2.symbol 
        and tf.rank - ty2.rank >=  8 
        and tf.rank - ty2.rank <= 11
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    data_merge_x_filter_0 as (
        select ty0.rank, ty0.symbol, ty0.datafqtr, ty0.rdq {merge_query_x_filter}
        from data0_filter ty0 inner join data1_filter ty1 inner join data2_filter ty2
        on ty0.symbol = ty1.symbol and ty0.rdq = ty1.rdq
        and ty0.symbol = ty2.symbol and ty0.rdq = ty2.rdq
    ), 
    data_merge_x_filter_1 as (
        select 'valid' as status, data_merge_x_filter_0.* from data_merge_x_filter_0
        {query_growth_filter}
    ), 
    filter2 as (
        with table_3_year_all as ( 
            select distinct t2.rank, t2.symbol, t2.rdq, t2.datafqtr
            from data_merge_x_filter_1 t1, filter_0 t2
            where t1.symbol = t2.symbol
            and julianday(t1.rdq) - julianday(t2.rdq) <= 1135
            and julianday(t1.rdq) - julianday(t2.rdq) >= -405
        )
        select t1.* from 
        filter_0 t1 inner join table_3_year_all t2
        on t1.symbol = t2.symbol and t1.rdq = t2.rdq
    ), 
    rdq_0_filter as (
        select distinct t2.rank, t2.symbol, t2.rdq, t2.datafqtr
        from data_merge_x_filter_1 t1, filter_0 t2
        where t1.symbol = t2.symbol
        and julianday(t1.rdq) - julianday(t2.rdq) <= 0
        and julianday(t1.rdq) - julianday(t2.rdq) >= -405 
    ),
    data0 as (
        select tf.rank, tf.symbol, tf.datafqtr, tf.rdq, {col_query_avg_0}
        from rdq_0_filter tf, filter2 ty0
        where tf.symbol = ty0.symbol 
        and tf.rank - ty0.rank >= 0 
        and tf.rank - ty0.rank <= 3 
        group by tf.symbol, tf.rdq, tf.datafqtr
    ), 
    data1 as (
        select tf.rank, tf.symbol, tf.rdq, {col_query_avg_1}
        from rdq_0_filter tf, filter2 ty1
        where tf.symbol = ty1.symbol
        and tf.rank - ty1.rank >= 4
        and tf.rank - ty1.rank <= 7
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    data2 as (
        select tf.rank, tf.symbol, tf.rdq, {col_query_avg_2}
        from rdq_0_filter tf, filter2 ty2
        where tf.symbol = ty2.symbol 
        and tf.rank - ty2.rank >=  8 
        and tf.rank - ty2.rank <= 11
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    dataq0 as (
        select tf.rank, tf.symbol, tf.rdq, {col_query_avg_q0}
        from rdq_0_filter tf, filter2 tq
        where tf.symbol = tq.symbol and tf.rank = tq.rank 
        group by tf.symbol, tf.rdq, tf.datafqtr
    ), 
    dataq1 as (
        select tf.rank, tf.symbol, tf.rdq, {col_query_avg_q1}
        from rdq_0_filter tf, filter2 tq
        where tf.symbol = tq.symbol and tf.rank - tq.rank = 1 
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    dataq4 as (
        select tf.rank, tf.symbol, tf.rdq, {col_query_avg_q4}
        from rdq_0_filter tf, filter2 tq
        where tf.symbol = tq.symbol and tf.rank - tq.rank = 4 
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    datapq1 as (
        select tf.symbol, tf.rdq, {col_query_avg_pq1}
        from rdq_0_filter tf, filter2 tq
        where tf.symbol = tq.symbol and tf.rank - tq.rank = -1 
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    datapq2 as (
        select tf.symbol, tf.rdq, {col_query_avg_pq2}
        from rdq_0_filter tf, filter2 tq
        where tf.symbol = tq.symbol and tf.rank - tq.rank = -2
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    datapq3 as (
        select tf.symbol, tf.rdq, {col_query_avg_pq3}
        from rdq_0_filter tf, filter2 tq
        where tf.symbol = tq.symbol and tf.rank - tq.rank = -3
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    datapq4 as (
        select tf.symbol, tf.rdq, {col_query_avg_pq4}
        from rdq_0_filter tf, filter2 tq
        where tf.symbol = tq.symbol and tf.rank - tq.rank = -4
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    data_merge_x as (
        select ty0.rank, ty0.symbol, ty0.datafqtr, ty0.rdq {merge_query_x}
        from data0 ty0 inner join data1 ty1 inner join data2 ty2
        inner join dataq0 tq0 inner join dataq1 tq1 inner join dataq4 tq4 
        on ty0.symbol = ty1.symbol and ty0.rdq = ty1.rdq
        and ty0.symbol = ty2.symbol and ty0.rdq = ty2.rdq
        and ty0.symbol = tq0.symbol and ty0.rdq = tq0.rdq
        and ty0.symbol = tq1.symbol and ty0.rdq = tq1.rdq
        and ty0.symbol = tq4.symbol and ty0.rdq = tq4.rdq
    ), 
    data_merge_xy_1 as (
        select tyx.* {dict_merge_query_y['pq1']}
        from data_merge_x tyx left join datapq1 typq1
        on tyx.symbol = typq1.symbol and tyx.rdq = typq1.rdq
    ), 
    data_merge_xy_2 as (
        select tyx.* {dict_merge_query_y['pq2']}
        from data_merge_xy_1 tyx left join datapq2 typq2
        on tyx.symbol = typq2.symbol and tyx.rdq = typq2.rdq
    ), 
    data_merge_xy_3 as (
        select tyx.* {dict_merge_query_y['pq3']}
        from data_merge_xy_2 tyx left join datapq3 typq3
        on tyx.symbol = typq3.symbol and tyx.rdq = typq3.rdq
    ), 
    data_merge_xy_4 as (
        select tyx.* {dict_merge_query_y['pq4']}
        from data_merge_xy_3 tyx left join datapq4 typq4
        on tyx.symbol = typq4.symbol and tyx.rdq = typq4.rdq
    ), 
    data_merge_xy as (
        select * from data_merge_xy_4
    ),
    data_translate as (
        select {query_translate} from data_merge_xy
    )
    select ty.status, tf.* from 
    data_translate tf left join data_merge_x_filter_1 ty
    on ty.rank = tf.rank
    order by rank
    """
    pd_data_raw = pd.read_sql(command_query, con)
    pd_data_raw = pd_data_raw.sort_values(by=['rdq_0', 'symbol'])
    #pd_data_raw_1 = pd_data_raw.copy()
    # print(command_query)
    print('Completed wharton financial report data pull')
    # Add marketcap info

    rdq_list = [i for i in pd_data_raw.columns if 'rdq' in i not in ['rdq_q4', 'rdq_q0']]
    pd_rdq_list = []
    for i_rdq, rdq in enumerate(rdq_list):
        pd_rdq_list.append(pd_data_raw[['symbol', rdq]].rename(columns={rdq: 'rdq'}))
    pd_rdq = pd.concat(pd_rdq_list).drop_duplicates().dropna().sort_values(by=['symbol', 'rdq'])

    if 1 == 0:
        _pd_marketcap_report = stock_price.get_marketcap_time(pd_rdq, time_col='rdq', avg=14)

        pd_data = pd_data_raw.copy()
        print('\rCompleted market report data pull')
        for i_rdq, rdq in enumerate(rdq_list):
            time_label = rdq.split('_')[-1]
            pd_data = pd_data.merge(_pd_marketcap_report, left_on=['symbol', rdq], right_on=['symbol', 'rdq'], how='left')
            pd_data = pd_data.drop_duplicates()
            pd_data = pd_data.rename(columns={'marketcap': f'marketcap_{time_label}'})
            pd_data = pd_data[[i for i in pd_data.columns if i != 'rdq']]

        keys_front = ['symbol', 'datafqtr'] + [f'marketcap_{i.split("_")[-1]}' for i in rdq_list]
        pd_data = pd_data[keys_front + [i for i in pd_data.columns if i not in keys_front]]
        pd_data = pd_data.sort_values(by=['rdq_0', 'symbol'])
        pd_data = pd_data.loc[~(pd_data.marketcap_0.isna() | pd_data.marketcap_1.isna() | pd_data.marketcap_2.isna())]
        pd_data_ori = pd_data.copy()
        print('Completed data aggregation')

if 'Define Function' == 'Define Function':
    def convert_decision_time(decision_time):
        if type(decision_time) in [float, int]:
            if decision_time < 1900:
                raise ValueError("decision_time has to be the recent year")
            else:
                _year, _days = int(decision_time // 1), (decision_time - decision_time // 1) * 361
                _month = str(max(int(_days // 30), 1)).rjust(2, '0')
                _day = str(max(round(int(_days % 30)), 1)).rjust(2, '0')
                decision_time_final = f'{_year}-{_month}-{_day}'
        else:
            _info = decision_time.split('-')
            if len(_info) == 1:
                decision_time_final = f'{_info[0]}-01-01'
            elif len(_info) == 2:
                decision_time_final = f'{_info[0]}-{_info[1]}-01'
            else:
                decision_time_final = decision_time
        return decision_time_final

    def plot_year_num_dist(pd_data):
        pd_data_plot = pd_data.dropna()[['symbol', 'rdq_p']].copy()
        pd_data_plot['year'] = pd_data_plot.rdq_p.str[:4].astype(int) + ((pd_data_plot.rdq_p.str[5:7].astype(int) - 1 - 1) // 3) / 4
        plt.hist(pd_data_plot['year'], bins=100)

    def plot_feature_dist(pd_mdata, features_x):
        fig, ax = plt.subplots(4, 5, figsize=(15, 7.5))
        ax = fig.axes
        for i, feature in enumerate(features_x):
            dist = pd_mdata[feature]
            ax[i].hist(dist, bins=35)
            ax[i].set_title(feature)
        fig.tight_layout()

    def y_transform(y, direction='encode', func_shift=1, func_power=2, dict_transform=None):
        if direction.lower() == 'encode':
            y_output = (y + func_shift) ** func_power
            median = np.quantile(y_output, 0.5)
            std = y_output.std()
            y_output = (y_output - median) / std / dict_transform['std_adjust']
            return y_output, median, std
        elif direction.lower() == 'decode':
            y_before_norm = y * dict_transform['y_std'] * dict_transform['std_adjust'] + dict_transform['y_median']
            y_output = (y_before_norm ** (1 / func_power)) - func_shift
            return y_output
        else:
            raise ValueError(f'direction can only be either encode or decode, input is {direction}')

    def prepare_features(pd_data, dict_transform, training=False):
        coeff_fade = dict_transform['coeff_fade']
        if training:
            p_feature = dict_transform['p_feature_train']
        else:
            p_feature = dict_transform['p_feature_decision']

        time_col = f'rdq_{p_feature.split("_")[-1]}'
        pd_mdata = pd_data[[time_col]].copy()
        if (coeff_fade > 0) & (coeff_fade <= 1):
            pd_mdata['weight'] = coeff_fade ** (np.abs(pd.to_datetime(pd_mdata[time_col]) -
                                                       pd.to_datetime(pd_mdata[time_col]).max()).dt.days / 365)
        else:
            raise ValueError('coeff_fade has to be a positive number that is smaller 1 ')
        pd_mdata['mc_growth'] = pd_data[p_feature] / pd_data['marketcap_0']

        pd_mdata['mc_growth_log'] = list(np.log10(pd_mdata['mc_growth']))
        pd_mdata['mc_growth_log_squred'] = list(pd_mdata['mc_growth_log'] ** 2)

        features_bvr_year = ['cur_asset', 'cur_liab', 'cash_invest', 'cash_flow', 'revenue', 'profit']
        features_growth = ['book_value', 'revenue']
        mc_bv_years = 3
        for mc_bv_year in range(mc_bv_years):
            pd_mdata[f'mc_bv_{mc_bv_year}'] = list(pd_data[f'marketcap_{mc_bv_year}'] / pd_data[f'book_value_{mc_bv_year}'])
        pd_mdata['mc_bv_q1'] = list(pd_data[f'marketcap_0'] / pd_data[f'book_value_q1'])
        pd_mdata['mc_bv_q4'] = list(pd_data[f'marketcap_0'] / pd_data[f'book_value_1'])

        features_x = [f'mc_bv_{mc_bv_year}' for mc_bv_year in range(mc_bv_years)] + ['mc_bv_q1', 'mc_bv_q4']

        for feature in features_growth:
            for i_year in range(n_year_x - 1):
                feature_x = f'{feature}_growth_{i_year}'
                pd_mdata[feature_x] = list(pd_data[f'{feature}_{i_year}'] / pd_data[f'{feature}_{i_year + 1}'])
                features_x.append(feature_x)

            for i_quarter in [1, 4]:
                feature_x = f'{feature}_growth_q{i_quarter}'
                pd_mdata[feature_x] = list(pd_data[f'{feature}_0'] / pd_data[f'{feature}_q{i_quarter}'])
                features_x.append(feature_x)

        for i_year in range(n_year_x):
            for feature in features_bvr_year:
                feature_x = f'bvr_{feature}_{i_year}'
                pd_mdata[feature_x] = list(pd_data[f'{feature}_{i_year}'] / pd_data[f'book_value_{i_year}'])
                features_x.append(feature_x)
        for i_quarter in [1, 4]:
            for feature in features_bvr_year:
                feature_x = f'bvr_{feature}_q{i_quarter}'
                pd_mdata[feature_x] = list(pd_data[f'{feature}_0'] / pd_data[f'book_value_q{i_quarter}'])
                features_x.append(feature_x)

        return pd_mdata, features_x

    def prepare_lstm_computing(pd_mdata, dict_transform, output_y=True):
        # features_bvr_year = ['cur_asset', 'cur_liab', 'cash_invest', 'cash_flow', 'revenue', 'profit']
        features_bvr_year = []
        features_growth = ['book_value', 'revenue']
        n_year_x = dict_transform['n_year_x']

        time_stamps = list(range(n_year_x))[:0:-1] + ['q4', 'q1', '0']
        x_array_list = []
        for time_stamp in time_stamps:
            _features = [f'bvr_{i}_{time_stamp}' for i in features_bvr_year] + [f'mc_bv_{time_stamp}'] + \
                        [f'{i}_growth_{time_stamp}' for i in features_growth]
            pd_array = pd_mdata[[_features[0]]].copy()
            for col in _features[1:]:
                if col in pd_mdata.columns:
                    pd_array[col] = pd_mdata[col]
                else:
                    pd_array[col] = 0
            x_array_list.append(pd_array.values)
        x_array = np.concatenate(x_array_list, axis=1)
        features_time_stamp = len(features_bvr_year) + len(features_growth) + 1
        x_train = x_array.reshape((len(x_array), len(time_stamps), features_time_stamp))
        if output_y:
            y_train_ori = pd_mdata['mc_growth_log'].values
            y_train, y_median, y_std = y_transform(y_train_ori, 'encode', func_shift, func_power, dict_transform)
            dict_transform['y_median'] = y_median
            dict_transform['y_std'] = y_std
        else:
            y_train = None

        return x_train, y_train, dict_transform

    def get_model_lstm(pd_data, dict_transform):

        aug_sigma = dict_transform['aug_sigma']
        pd_mdata, features_x = prepare_features(pd_data, dict_transform['p_feature'], training=True)

        for feature in features_x:
            col = np.log10(pd_mdata[feature])
            mean, std = col.mean(), col.std()
            dict_transform['mean'][feature] = mean
            dict_transform['std'][feature] = std
            col = (col - mean) / std / dict_transform['std_adjust']
            pd_mdata[feature] = col

        pd_mdata_cal = pd_mdata

        if dict_transform['aug_size'] > 0:
            n_extra = aug_size * len(pd_mdata_cal)
            pd_mdata_cal_aug = pd.concat([pd_mdata_cal for _ in range(int(np.ceil(dict_transform['aug_size'], )))])
            pd_mdata_cal_aug = pd_mdata_cal_aug.iloc[:n_extra].copy()
            for feature in features_x:
                coeff = np.random.randn(len(pd_mdata_cal_aug)) * aug_sigma / dict_transform['std_adjust']
                pd_mdata_cal_aug[feature] = pd_mdata_cal_aug[feature] + coeff
            pd_mdata_cal = pd.concat([pd_mdata_cal, pd_mdata_cal_aug])

        x_train, y_train, dict_transform = prepare_lstm_computing(pd_mdata_cal, dict_transform)

        layer_input = tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))
        # Shape [batch, time, features] => [batch, time, lstm_units]
        x = tf.keras.layers.LSTM(dict_transform['lstm_units'])(layer_input)
        x = tf.keras.layers.Dense(units=1)(x)
        regr = tf.keras.models.Model(inputs=[layer_input], outputs=x)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='mean_absolute_error', patience=5, mode='min')

        regr.compile(loss=tf.losses.MeanSquaredError(),
                           optimizer=tf.optimizers.Adam(),
                           metrics=[tf.metrics.MeanAbsoluteError()])

        _ = regr.fit(x_train, y_train, epochs=dict_transform['lstm_epochs'], batch_size=128, callbacks=[early_stopping])

        regr_list = [regr]
        dict_transform['features_x'] = features_x
        return dict_transform, regr_list

    def get_model_sklearn(pd_data, dict_transform, n_estimators=500, learning_rate=1, max_depth=3,
                          tree_method=None, predictor=None):

        n_estimators_list = n_estimators if type(n_estimators) in [list, np.ndarray, range] else [n_estimators]
        learning_rate_list = learning_rate if type(learning_rate) in [list, np.ndarray, range] else [learning_rate]

        func_shift, func_power = dict_transform['func_shift'], dict_transform['func_power']
        aug_sigma = dict_transform['aug_sigma']

        pd_mdata, features_x = prepare_features(pd_data, dict_transform, training=True)

        features_bvr_year = ['cur_asset', 'cash_flow', 'revenue', 'profit']
        features_growth = ['book_value', 'revenue']
        features_x_select = ['mc_bv_0', 'mc_bv_1']
        for _ in features_growth:
            # features_x_select += [i for i in features_x if (_ in i) & ('growth' in i) & ('q4' not in i)]
            # features_x_select += [i for i in features_x if (_ in i) & ('growth' in i)]
            features_x_select += [i for i in features_x if (_ in i) & ('growth' in i)]
        for _ in features_bvr_year:
            features_x_select += [i for i in features_x if (_ in i) & ('bvr' in i) & (('0' in i) | ('0' in i))]
        for feature in features_x:
            col = np.log10(pd_mdata[feature])
            mean, std = col.mean(), col.std()
            dict_transform['mean'][feature] = mean
            dict_transform['std'][feature] = std
            col = (col - mean) / std / dict_transform['std_adjust']
            pd_mdata[feature] = col

        pd_mdata_cal = pd_mdata

        if dict_transform['aug_size'] > 0:
            n_extra = aug_size * len(pd_mdata_cal)
            pd_mdata_cal_aug = pd.concat([pd_mdata_cal for _ in range(int(np.ceil(dict_transform['aug_size'], )))])
            pd_mdata_cal_aug = pd_mdata_cal_aug.iloc[:n_extra].copy()
            for feature in features_x:
                coeff = np.random.randn(len(pd_mdata_cal_aug)) * aug_sigma / dict_transform['std_adjust']
                pd_mdata_cal_aug[feature] = pd_mdata_cal_aug[feature] + coeff
            pd_mdata_cal = pd.concat([pd_mdata_cal, pd_mdata_cal_aug])

        # regr = RandomForestRegressor(max_depth=3, n_estimators=2500)
        # regr = GradientBoostingRegressor(max_depth=3, n_estimators=n_estimators)
        regr_list = []
        x_train, y_train_ori = pd_mdata_cal[features_x_select].values, pd_mdata_cal['mc_growth_log'].values
        y_train, y_median, y_std = y_transform(y_train_ori, 'encode', func_shift, func_power, dict_transform)
        weight_train = pd_mdata_cal['weight']
        dict_transform['y_median'] = y_median
        dict_transform['y_std'] = y_std
        time_start = time.time()
        for i_regr in range(len(n_estimators_list)):
            n_estimators = n_estimators_list[i_regr]
            learning_rate = learning_rate_list[i_regr]
            regr = xgboost.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                        predictor=predictor, tree_method=tree_method, random_state=np.random.randint(999999))
            regr.fit(x_train, y_train, sample_weight=weight_train)
            regr_list.append(regr)
            time_span = round(time.time() - time_start, 1)
            print(f'\rCompleted regression {i_regr + 1}/{len(n_estimators_list)} - Time {time_span} s', end='')
        print()
        dict_transform['features_x'] = features_x
        dict_transform['features_x_select'] = features_x_select
        return dict_transform, regr_list

    def get_prediction(pd_data, dict_transform, regr_list):
        if type(regr_list) is list:
            regr_list = regr_list
        else:
            regr_list = [regr_list]
        # n_year_x = dict_transform['n_year_x']
        # features_x = dict_transform['features_x']
        func_shift, func_power = dict_transform['func_shift'], dict_transform['func_power']
        pd_mdata, features_x = prepare_features(pd_data, dict_transform)

        for feature in dict_transform['mean']:
            col = np.log10(pd_mdata[feature])
            mean, std = dict_transform['mean'][feature], dict_transform['std'][feature]
            col = (col - mean) / std / dict_transform['std_adjust']
            pd_mdata[feature] = col

        X_cal = pd_mdata[features_x].values

        # y_ori = pd_mdata['mc_growth_log'].values
        # y_actual, y_median, y_std = y_transform(y_ori, 'encode', func_shift, func_power, dict_transform)
        if dict_transform['aug_size'] > 0:
            x_array_aug = np.concatenate([X_cal for _ in range(dict_transform['aug_size'])], axis=0)
            coeff_array = np.random.randn(x_array_aug.shape[0], x_array_aug.shape[1]) * aug_sigma / dict_transform['std_adjust']
            x_array_aug = x_array_aug + coeff_array
            x_array_pred = np.concatenate([X_cal, x_array_aug])
        else:
            x_array_pred = X_cal
        y_array_pred_list = []
        for i_regr, regr in enumerate(regr_list):
            _pd_mdata_pred = pd.DataFrame(data=x_array_pred, columns=features_x)
            if 'keras' in str(type(regr)):
                x_array_final, _, _ = prepare_lstm_computing(_pd_mdata_pred, dict_transform, output_y=False)
            else:
                x_array_final = _pd_mdata_pred[dict_transform['features_x_select']]
            y_array_as_pred = regr.predict(x_array_final)
            y_array_pred_entry = y_array_as_pred.reshape(dict_transform['aug_size'] + 1, X_cal.shape[0]).T
            y_array_pred_list.append(y_array_pred_entry)
        y_pred_concat_as_pred = np.concatenate(y_array_pred_list, axis=1)
        y_pred_concat = y_transform(y_pred_concat_as_pred, 'decode', func_shift, func_power, dict_transform)
        y_pred_mean = y_pred_concat.mean(axis=1)
        y_pred_median = np.quantile(y_pred_concat, 0.5, axis=1)
        y_pred_std = y_pred_concat.std(axis=1)
        return y_pred_mean, y_pred_median, y_pred_std

    def e2e_pred_data(pd_data, dict_transform, regr_list, n_sigma=2):
        log_grow_pred_mean, log_grow_pred_median, log_grow_pred_std = get_prediction(pd_data, dict_transform, regr_list)
        head_keys = ['symbol', 'datafqtr', dict_transform['p_feature_decision'], 'marketcap_0', 'log_growth_mc', 'log_growth_mc_pred_min',
                     'log_growth_mc_pred_median', 'log_growth_mc_pred_mean', 'log_growth_mc_pred_std']
        pd_data['log_growth_mc_pred_mean'] = log_grow_pred_mean
        pd_data['log_growth_mc_pred_median'] = log_grow_pred_median
        pd_data['log_growth_mc_pred_std'] = log_grow_pred_std
        pd_data['log_growth_mc_pred_min'] = log_grow_pred_mean - log_grow_pred_std * n_sigma

        pd_data['log_growth_mc'] = np.log10(pd_data[dict_transform['p_feature_decision']] / pd_data['marketcap_0'])
        pd_data = pd_data[head_keys + [i for i in pd_data.columns if i not in head_keys]]
        return pd_data


if 1 == 1:
    pd_data = pd_data_ori.copy()
    decision_time = 2014

    decision_time_final = convert_decision_time(decision_time)


if 'Training' == 'Training0':
    predict_method = 'sklearn'
    dict_revenue_growth_min = {'1': 0.0, '0': 0.1}
    dict_book_value_growth_min = {'1': 0.0, '0': 0.1}
    dict_revenue_growth_max = {}
    dict_book_value_growth_max = {}
    mc_book_ratio = [1, 65]
    mc_revenue_ratio = [1, 65]
    ratio_train = 2014
    coeff_fade = 0.8
    func_shift, func_power = 5, 3
    std_adjust = 2
    aug_size, aug_sigma = 100, 0.2
    time_shuffle = 'time'
    p_feature_train = 'marketcap_p'
    p_feature_decision = 'marketcap_p'

    #################################################
    # sklearn parameters
    lstm_units = 32
    epochs = 100
    #################################################
    # sklearn parameters
    n_regr = 30
    n_estimators_min, n_estimators_max = 100, 150
    learning_rate_min, learning_rate_max = 0.85, 1
    #################################################

    n_year_x = 3
    marketcap_min = 100
    max_depth, tree_method, predictor = 3, 'gpu_hist', 'gpu_predictor'
    n_estimators_list = range(n_estimators_min, n_estimators_max, (n_estimators_max - n_estimators_min) // n_regr)
    learning_rate_list = np.arange(learning_rate_min, learning_rate_max, (learning_rate_max - learning_rate_min) / n_regr)
    n_estimators_list = list(n_estimators_list) + list(n_estimators_list)
    learning_rate_list = list(learning_rate_list) + list(learning_rate_list)[::-1]
    _ = min(len(n_estimators_list), len(learning_rate_list))
    n_estimators_list, learning_rate_list = n_estimators_list[:_], learning_rate_list[:_]

    _pd_data = pd_data_ori.copy()
    if time_shuffle.lower == 'random':
        _pd_data = _pd_data.iloc[sorted(np.arange(10), key=lambda x: np.random.random())]
    for i in dict_revenue_growth_min:
        _pd_data = _pd_data.loc[(_pd_data[f'revenue_{i}'] / _pd_data[f'revenue_{int(i) + 1}']) > (1 + dict_revenue_growth_min[i])]
    for i in dict_book_value_growth_min:
        _pd_data = _pd_data.loc[(_pd_data[f'book_value_{i}'] / _pd_data[f'book_value_{int(i)+1}']) > (1 + dict_book_value_growth_min[i])]
    for i in dict_revenue_growth_max:
        _pd_data = _pd_data.loc[(_pd_data[f'revenue_{i}'] / _pd_data[f'revenue_{int(i) + 1}']) <= (1 + dict_revenue_growth_max[i])]
    for i in dict_book_value_growth_max:
        _pd_data = _pd_data.loc[(_pd_data[f'book_value_{i}'] / _pd_data[f'book_value_{int(i)+1}']) <= (1 + dict_book_value_growth_max[i])]

    #_pd_data = _pd_data.loc[(_pd_data[f'revenue_0'] / _pd_data[f'revenue_1']) >= (_pd_data[f'revenue_1'] / _pd_data[f'revenue_2'])]
    #_pd_data = _pd_data.loc[(_pd_data[f'book_value_0'] / _pd_data[f'book_value_1']) >=
    #                        (_pd_data[f'book_value_1'] / _pd_data[f'book_value_2'])]
    _pd_data = _pd_data.loc[(_pd_data[f'revenue_q0'] / _pd_data[f'revenue_q4']) >= (_pd_data[f'revenue_1'] / _pd_data[f'revenue_2'])]
    _pd_data = _pd_data.loc[(_pd_data[f'book_value_q0'] / _pd_data[f'book_value_q4']) >=
                            (_pd_data[f'book_value_1'] / _pd_data[f'book_value_2'])]
    #_pd_data = _pd_data.loc[(_pd_data[f'revenue_q0'] / _pd_data[f'revenue_q4']) *
    #                        (_pd_data[f'book_value_q0'] / _pd_data[f'book_value_q4']) >=
    #                        (_pd_data[f'revenue_1'] / _pd_data[f'revenue_2']) *
    #                        (_pd_data[f'book_value_1'] / _pd_data[f'book_value_2'])]
    _pd_data = _pd_data.loc[((_pd_data[p_feature_train] / _pd_data.marketcap_0) <= 10) &
                            ((_pd_data[p_feature_train] / _pd_data.marketcap_0) >= 10 ** -1)]
    _pd_data = _pd_data.loc[((_pd_data.marketcap_0 / _pd_data.book_value_0) <= mc_book_ratio[1]) &
                            ((_pd_data.marketcap_0 / _pd_data.book_value_0) >= mc_book_ratio[0])]
    _pd_data = _pd_data.loc[((_pd_data.marketcap_0 / _pd_data.revenue_0) <= mc_revenue_ratio[1]) &
                            ((_pd_data.marketcap_0 / _pd_data.revenue_0) >= mc_revenue_ratio[0])]
    _pd_data = _pd_data.loc[_pd_data.marketcap_0 >= marketcap_min]
    _pd_data = _pd_data.sort_values(by='rdq_0')

    _ = p_feature_decision.split('_')[-1]
    time_label_predict = p_feature_decision.split('_')[-1]
    rdq_predict = f'rdq_{time_label_predict}'

    _pd_data['datafqtr'] = (_pd_data[rdq_predict].str[:4].astype(float) + (_pd_data[rdq_predict].str[5:7].astype(float) - 1) / 12 +
                            (_pd_data[rdq_predict].str[8:10].astype(float) - 1) / 365).round(2)
    time_label_predict_int = 4 if 'q' not in _ else int(_[-1])


    pd_data = _pd_data.copy()
    pd_data_extra = _pd_data.loc[(_pd_data.num_p < time_label_predict_int) | (_pd_data.num_p.isna())].copy()
    pd_data.index = np.arange(len(pd_data))

    if type(ratio_train) in [float, int]:
        if ratio_train < 1:
            n_threshold_train = int(len(pd_data) * ratio_train)
        else:
            n_threshold_train = len(pd_data.loc[pd_data.datafqtr <= ratio_train])
    else:
        if '-' in ratio_train:
            n_threshold_train = len(pd_data.loc[pd_data[rdq_predict] <= str(ratio_train)])
        else:
            n_threshold_train = len(pd_data.loc[pd_data.datafqtr <= ratio_train])

    pd_data_train = pd_data.iloc[:n_threshold_train].copy()

    regr_list = []
    dict_transform = {'mean': {}, 'std': {}, 'n_year_x': n_year_x, 'func_shift': func_shift, 'func_power': func_power,
                      'aug_size': aug_size, 'aug_sigma': aug_sigma, 'std_adjust': std_adjust, 'lstm_units': lstm_units,
                      'lstm_epochs': epochs, 'p_feature_train': p_feature_train, 'p_feature_decision': p_feature_decision,
                      'coeff_fade': coeff_fade}
    if predict_method.lower() == 'sklearn':
        dict_transform, regr_list = get_model_sklearn(pd_data_train, dict_transform, n_estimators=n_estimators_list,
                                                      learning_rate=learning_rate_list, max_depth=max_depth, tree_method=tree_method,
                                                      predictor=predictor)
    elif predict_method.lower() == 'lstm':
        dict_transform, regr_list = get_model_lstm(pd_data_train, dict_transform)
    else:
        raise ValueError('predict_method can only be in [lstm, sklearn]')

    _pd_data = e2e_pred_data(_pd_data.copy(), dict_transform, regr_list)
    pd_data_train_show = _pd_data.iloc[:n_threshold_train].copy()
    time_max_train = pd_data_train_show[rdq_predict].max()

    _pd_data_show = _pd_data.loc[_pd_data.rdq_0 >= time_max_train]
    pd_data_show = _pd_data_show.loc[_pd_data_show.num_p >= time_label_predict_int].copy()
    pd_data_extra_show = _pd_data_show.loc[(_pd_data_show.num_p < time_label_predict_int) | (_pd_data_show.num_p.isna())].copy()

    time_start_decision = pd_data_show[rdq_predict].min()
    n_years = round((pd.to_datetime(pd_data_show[rdq_predict].max()) - pd.to_datetime(time_start_decision)).days / 365)
    dict_pd_data = {}
    for i_year in range(n_years):
        time_year_min = str(pd.to_datetime(time_start_decision) + pd.to_timedelta(f'{365 * i_year} day'))[:10]
        time_year_max = str(pd.to_datetime(time_start_decision) + pd.to_timedelta(f'{365 * (i_year + 1)} day'))[:10]
        pd_data_year_entry = pd_data_show.loc[(pd_data_show[rdq_predict] > time_year_min) &
                                              (pd_data_show[rdq_predict] <= time_year_max)]
        dict_pd_data[f'{i_year + 1}_year'] = pd_data_year_entry
    dict_pd_data['test_extra'] = pd_data_extra_show.loc[pd_data_extra_show.num_p < time_label_predict_int]

    #metric_plot = 'log_growth_mc_pred_min'
    metric_plot = 'log_growth_mc_pred_median'
    #metric_plot = 'log_growth_mc_pred_mean'

    def get_summary(dict_pd_data, top_p=0.97):
        dict_sum = {'data_label': [], 'p_value': [], 'mean_all': [], 'mean_top': [], 'std_all': [], 'std_top': [],
                    'num_all': [], 'num_top': [], 'profit_all': [], 'profit_top': [], 'time_min_buy': [], 'time_max_buy': [],
                    'criteria': []}
        for label in dict_pd_data:
            pd_data = dict_pd_data[label]
            if len(pd_data) > 0:
                pd_top = pd_data.loc[pd_data[metric_plot] >= pd_data[metric_plot].quantile(top_p)]
                array_all, array_top = pd_data['log_growth_mc'], pd_top['log_growth_mc']
                p_value = round(scipy.stats.f_oneway(array_all, array_top).pvalue, 4)
                mean_all, mean_top = array_all.mean(), array_top.mean()
                std_all, std_top = array_all.std(), array_top.std()
                num_all, num_top = len(array_all), len(array_top)
                dict_sum['data_label'].append(label)
                dict_sum['p_value'].append(p_value)
                dict_sum['mean_all'].append(mean_all)
                dict_sum['mean_top'].append(mean_top)
                dict_sum['std_all'].append(std_all)
                dict_sum['std_top'].append(std_top)
                dict_sum['num_all'].append(num_all)
                dict_sum['num_top'].append(num_top)
                dict_sum['profit_all'].append(round((10 ** array_all).mean() - 1, 4))
                dict_sum['profit_top'].append(round((10 ** array_top).mean() - 1, 4))
                dict_sum['time_min_buy'].append(pd_data.rdq_0.min())
                dict_sum['time_max_buy'].append(pd_data.rdq_0.max())
                dict_sum['criteria'].append(pd_top[metric_plot].min())
        _pd_summary = pd.DataFrame(dict_sum)
        return _pd_summary

    top_p = 0.9
    pd_summary = get_summary(dict_pd_data, top_p=top_p)
    print(pd_summary)

# con.close()
