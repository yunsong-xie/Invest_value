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
import multiprocessing as mp


pd.set_option('display.max_column', 75)
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

if __name__ == '__main__0':

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

    symbols = ['AAPL', 'XOM', 'WMT', 'WERN', 'STE', 'HLT', 'CTXS', 'CTXS', 'KLAC']
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

    query_translate = ' rank, symbol, datafqtr, '
    for key_time in year_list + quarter_list + ['pq1', 'pq2', 'pq3', 'pq4']:
        if 'pq' not in str(key_time):
            for col in col_output_list_avg:
                if col in dict_col_name_reverse:
                    query_translate += f'{col}_{key_time} as {desc_output_dict[dict_col_name_reverse[col]]}_{key_time}, '
                else:
                    query_translate += f'{col}_{key_time} as {desc_output_dict[col]}_{key_time}, '
            query_translate += '\n'
        else:
            query_translate += f'rdq_{key_time}, '

    query_translate = query_translate[:-2]

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
            and julianday(t1.rdq) - julianday(t2.rdq) >= -770
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
        select tf.rank, tf.symbol, tf.datafqtr, tf.rdq, {col_query_avg_1}
        from rdq_0_filter tf, filter2 ty1
        where tf.symbol = ty1.symbol
        and tf.rank - ty1.rank >= 4
        and tf.rank - ty1.rank <= 7
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    data2 as (
        select tf.rank, tf.symbol, tf.datafqtr, tf.rdq, {col_query_avg_2}
        from rdq_0_filter tf, filter2 ty2
        where tf.symbol = ty2.symbol 
        and tf.rank - ty2.rank >=  8 
        and tf.rank - ty2.rank <= 11
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    dataq0 as (
        select tf.rank, tf.symbol, tf.datafqtr, tf.rdq, {col_query_avg_q0}
        from rdq_0_filter tf, filter2 tq
        where tf.symbol = tq.symbol and tf.rank = tq.rank 
        group by tf.symbol, tf.rdq, tf.datafqtr
    ), 
    dataq1 as (
        select tf.rank, tf.symbol, tf.datafqtr, tf.rdq, {col_query_avg_q1}
        from rdq_0_filter tf, filter2 tq
        where tf.symbol = tq.symbol and tf.rank - tq.rank = 1 
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    dataq4 as (
        select tf.rank, tf.symbol, tf.datafqtr, tf.rdq, {col_query_avg_q4}
        from rdq_0_filter tf, filter2 tq
        where tf.symbol = tq.symbol and tf.rank - tq.rank = 4 
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    datapq1 as (
        select tf.symbol, tf.datafqtr, tf.rdq, max(tq.rdq) as rdq_pq1
        from rdq_0_filter tf, filter2 tq
        where tf.symbol = tq.symbol and tf.rank - tq.rank = -1 
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    datapq2 as (
        select tf.symbol, tf.datafqtr, tf.rdq, max(tq.rdq) as rdq_pq2
        from rdq_0_filter tf, filter2 tq
        where tf.symbol = tq.symbol and tf.rank - tq.rank = -2
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    datapq3 as (
        select tf.symbol, tf.datafqtr, tf.rdq, max(tq.rdq) as rdq_pq3
        from rdq_0_filter tf, filter2 tq
        where tf.symbol = tq.symbol and tf.rank - tq.rank = -3
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    datapq4 as (
        select tf.symbol, tf.datafqtr, tf.rdq, max(tq.rdq) as rdq_pq4
        from rdq_0_filter tf, filter2 tq
        where tf.symbol = tq.symbol and tf.rank - tq.rank = -4
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    data_merge_x as (
        select ty0.rank, ty0.symbol, ty0.datafqtr, ty0.rdq {merge_query_x}
        from data0 ty0 inner join data1 ty1 inner join data2 ty2
        inner join dataq0 tq0 inner join dataq1 tq1 inner join dataq4 tq4 
        on ty0.symbol = ty1.symbol and ty0.rdq = ty1.rdq and ty0.datafqtr = ty1.datafqtr
        and ty0.symbol = ty2.symbol and ty0.rdq = ty2.rdq and ty0.datafqtr = ty2.datafqtr
        and ty0.symbol = tq0.symbol and ty0.rdq = tq0.rdq and ty0.datafqtr = tq0.datafqtr
        and ty0.symbol = tq1.symbol and ty0.rdq = tq1.rdq and ty0.datafqtr = tq1.datafqtr
        and ty0.symbol = tq4.symbol and ty0.rdq = tq4.rdq and ty0.datafqtr = tq4.datafqtr
    ), 
    data_merge_xy_1 as (
        select tyx.*, typq1.rdq_pq1
        from data_merge_x tyx left join datapq1 typq1
        on tyx.symbol = typq1.symbol and tyx.rdq = typq1.rdq and tyx.datafqtr = typq1.datafqtr
    ), 
    data_merge_xy_2 as (
        select tyx.*, typq2.rdq_pq2
        from data_merge_xy_1 tyx left join datapq2 typq2
        on tyx.symbol = typq2.symbol and tyx.rdq = typq2.rdq and tyx.datafqtr = typq2.datafqtr
    ), 
    data_merge_xy_3 as (
        select tyx.*, typq3.rdq_pq3
        from data_merge_xy_2 tyx left join datapq3 typq3
        on tyx.symbol = typq3.symbol and tyx.rdq = typq3.rdq and tyx.datafqtr = typq3.datafqtr
    ), 
    data_merge_xy_4 as (
        select tyx.*, typq4.rdq_pq4
        from data_merge_xy_3 tyx left join datapq4 typq4
        on tyx.symbol = typq4.symbol and tyx.rdq = typq4.rdq and tyx.datafqtr = typq4.datafqtr
    ), 
    data_merge_xy as (
        select * from data_merge_xy_4
    ),
    data_translate as (
        select {query_translate} from data_merge_xy
    ),
    data_final as ( 
        select ty.status, tf.* from 
        data_translate tf left join data_merge_x_filter_1 ty
        on ty.rank = tf.rank
        order by rank
    )
    select * from data_final
    """
    pd_data_raw = pd.read_sql(command_query, con)
    #pd_data_raw_1 = pd_data_raw.copy()
    # print(command_query)
    print('Completed Wharton financial report data pull')
    # Add marketcap info

    rdq_list = [i for i in pd_data_raw.columns if 'rdq' in i not in ['rdq_q4', 'rdq_q0']]
    pd_rdq_list = []
    for i_rdq, rdq in enumerate(rdq_list):
        pd_rdq_list.append(pd_data_raw[['symbol', rdq]].rename(columns={rdq: 'rdq'}))
    pd_rdq = pd.concat(pd_rdq_list).drop_duplicates().dropna().sort_values(by=['symbol', 'rdq'])

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
    pd_data = pd_data.loc[(~pd_data.rdq_pq1.isna()) | (pd_data.rdq_0 >= common_func.date(-90))]
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

    def prepage_training_data(pd_data_train):
        pd_data_train['num_min'] = (pd_data_train[['num_valid', 'num']].values.min(axis=1) * 4).astype(int)
        pd_train_list = []
        for i_num in range(4):
            num_min = i_num + 1
            pd_num = pd_data_train.loc[pd_data_train['num_min'] == num_min]
            for j_num in range(i_num + 1):
                pd_temp = pd_num.copy()
                pd_temp['marketcap_p'] = pd_temp[f'marketcap_pq{j_num + 1}']
                pd_temp['rdq_p'] = pd_temp[f'rdq_pq{j_num + 1}']
                pd_temp['num_p'] = j_num + 1
                pd_train_list.append(pd_temp)
        pd_train = pd.concat(pd_train_list)
        keys_remove = ['num_min']
        keys_head = ['datatype', 'symbol', 'datafqtr', 'num_p']
        pd_train = pd_train[keys_head + [i for i in pd_train.columns if i not in keys_remove + keys_head]]
        pd_train['num_p'] = pd_train['num_p'] / 4
        return pd_train

    def prepare_features(pd_data, dict_transform, data_type='training'):
        coeff_fade = dict_transform['coeff_fade']
        if data_type == 'training':
            p_feature = 'marketcap_p'
        elif data_type == 'evaluation':
            p_feature = dict_transform['p_feature_evaluation']
        elif data_type == 'investing':
            p_feature = None
        else:
            raise ValueError('datatype has to be in [training, evaluation, investing]')

        if p_feature:
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
        else:
            pd_mdata = pd_data[[]].copy()

        features_bvr_year = ['cur_asset', 'cur_liab', 'cash_invest', 'cash_flow', 'revenue', 'profit']
        features_growth = ['book_value', 'revenue']
        mc_bv_years = 3
        for mc_bv_year in range(mc_bv_years):
            pd_mdata[f'mc_bv_{mc_bv_year}'] = list(pd_data[f'marketcap_{mc_bv_year}'] / pd_data[f'book_value_{mc_bv_year}'])
        pd_mdata['mc_bv_q1'] = list(pd_data[f'marketcap_0'] / pd_data[f'book_value_q1'])
        pd_mdata['mc_bv_q4'] = list(pd_data[f'marketcap_0'] / pd_data[f'book_value_1'])
        pd_mdata['num'] = pd_data['num']
        pd_mdata['num_p'] = pd_data['num_p']

        features_x = [i for i in pd_mdata.columns if 'mc_bv' in i] + ['num_p']

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

    def get_model_sklearn(pd_data, dict_transform):

        func_shift, func_power = dict_transform['func_shift'], dict_transform['func_power']
        aug_sigma = dict_transform['aug_sigma']

        pd_mdata, features_x = prepare_features(pd_data, dict_transform, data_type='training')

        features_bvr_year = ['cur_asset', 'cash_flow', 'revenue', 'profit']
        features_growth = ['book_value', 'revenue']
        features_x_select = ['mc_bv_0', 'mc_bv_1', 'num_p']

        features_bvr_year = ['cur_asset', 'cash_flow', 'revenue', 'profit']
        features_growth = ['book_value', 'revenue']
        features_x_select = ['num_p', 'mc_bv_0']
        features_growth_time_label = ['year']

        for _ in features_growth:
            # features_x_select += [i for i in features_x if (_ in i) & ('growth' in i) & ('q4' not in i)]
            # features_x_select += [i for i in features_x if (_ in i) & ('growth' in i)]
            if 'year' in features_growth_time_label:
                features_x_select += [i for i in features_x if (_ in i) & ('growth' in i) & ('q' not in i) & ('0' in i)]
            if 'quarter' in features_growth_time_label:
                features_x_select += [i for i in features_x if (_ in i) & ('growth' in i) & ('q' in i) & ('1' in i)]
        for _ in features_bvr_year:
            features_x_select += [i for i in features_x if (_ in i) & ('bvr' in i) & (('0' in i) | ('0' in i))]
        for feature in features_x:
            if feature in dict_transform['features_exempt']:
                dict_transform['mean'][feature] = 0
                dict_transform['std'][feature] = 1
            else:
                ind_neg = pd_mdata[feature] <= 0
                if any(ind_neg):
                    pd_mdata.loc[ind_neg, feature] = pd_mdata.loc[pd_mdata[feature] > 0, feature].min()
                col = np.log10(pd_mdata[feature].values)

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
                if feature not in dict_transform['features_exempt']:
                    coeff = np.random.randn(len(pd_mdata_cal_aug)) * aug_sigma / dict_transform['std_adjust']
                    pd_mdata_cal_aug[feature] = pd_mdata_cal_aug[feature] + coeff
            pd_mdata_cal = pd.concat([pd_mdata_cal, pd_mdata_cal_aug])

        x_train, y_train_ori = pd_mdata_cal[features_x_select].values, pd_mdata_cal['mc_growth_log'].values
        y_train, y_median, y_std = y_transform(y_train_ori, 'encode', func_shift, func_power, dict_transform)
        weight_train = pd_mdata_cal['weight'].values
        dict_transform['y_median'] = y_median
        dict_transform['y_std'] = y_std

        pd_estimator = pd.DataFrame({'estimator': dict_transform['n_estimators_list'],
                                     'learning_rate': dict_transform['learning_rates']})
        pd_estimator['state'] = np.random.randint(9999999, size=len(pd_estimator))
        pd_estimator = pd_estimator.iloc[sorted(range(len(pd_estimator)), key=lambda x: np.random.random())]

        dict_regr_parameter = {i: dict_transform[i] for i in ['max_depth', 'tree_method', 'predictor', 'booster', 'subsample']}

        if dict_transform['n_threads'] > 1:
            n_threads = dict_transform['n_threads']
            mp_queue = mp.Queue()

            for i_thread in range(n_threads):
                pd_estimator_thread = pd_estimator.iloc[i_thread: (i_thread + len(pd_estimator) // n_threads)]
                mp.Process(target=sklearn_training_thread, args=(pd_estimator_thread, x_train, y_train, weight_train,
                                                                 dict_regr_parameter, mp_queue)).start()

            return_count, regr_list, time_start = 0, [], time.time()

            while return_count < n_threads:
                time.sleep(0.1)
                if not mp_queue.empty():
                    _result = mp_queue.get()
                    regr_list += _result
                    return_count += 1
                time_span = time.time() - time_start
                if time_span > 180:
                    raise ChildProcessError('Something went wrong')
        else:
            pd_estimator_thread = pd_estimator
            regr_list = sklearn_training_thread(pd_estimator_thread, x_train, y_train, weight_train, dict_regr_parameter)

        dict_transform['features_x'] = features_x
        dict_transform['features_x_select'] = features_x_select
        return dict_transform, regr_list

    def sklearn_training_thread(pd_estimator_thread, x_train, y_train, weight_train, dict_regr_parameter, mp_queue=None):

        max_depth, tree_method = dict_regr_parameter['max_depth'], dict_regr_parameter['tree_method']
        predictor = dict_regr_parameter['predictor']
        booster, subsample = dict_regr_parameter['booster'], dict_regr_parameter['subsample']

        regr_list = []

        time_start = time.time()
        for i_regr in range(len(pd_estimator_thread)):
            n_estimators = int(pd_estimator_thread.iloc[i_regr]['estimator'])
            learning_rate = pd_estimator_thread.iloc[i_regr]['learning_rate']
            state = int(pd_estimator_thread.iloc[i_regr]['state'])
            if dict_transform['regr_type'] == 'GB':
                regr = xgboost.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                            predictor=predictor, tree_method=tree_method, random_state=state)
            elif dict_transform['regr_type'] == 'RF':
                regr = xgboost.XGBRFRegressor(n_estimators=n_estimators, num_parallel_tree=n_estimators, subsample=subsample,
                                              max_depth=max_depth, learning_rate=learning_rate, booster=booster,
                                              predictor=predictor, random_state=state, n_jobs=6)
            else:
                raise KeyError('regr_type can only be [GB, RF]')
            regr.fit(x_train, y_train, sample_weight=weight_train)
            regr_list.append(regr)

            time_span = round(time.time() - time_start, 1)
            print(f'\rCompleted regression {i_regr + 1}/{len(pd_estimator_thread)} - Time {time_span} s', end='')
        print()

        if mp_queue is None:
            return regr_list
        else:
            mp_queue.put(regr_list)

    def get_prediction(pd_data, dict_transform, regr_list):
        if type(regr_list) is list:
            regr_list = regr_list
        else:
            regr_list = [regr_list]
        # n_year_x = dict_transform['n_year_x']
        # features_x = dict_transform['features_x']
        func_shift, func_power = dict_transform['func_shift'], dict_transform['func_power']
        pd_mdata, features_x = prepare_features(pd_data, dict_transform, data_type='investing')

        for feature in dict_transform['mean']:
            ind_neg = pd_mdata[feature] < 0
            if any(ind_neg):
                pd_mdata.loc[ind_neg, feature] = pd_mdata.loc[pd_mdata[feature] > 0, feature].min()
            col = np.log10(pd_mdata[feature].values)
            mean, std = dict_transform['mean'][feature], dict_transform['std'][feature]
            col = (col - mean) / std / dict_transform['std_adjust']
            pd_mdata[feature] = col

        X_cal = pd_mdata[features_x].values

        # y_ori = pd_mdata['mc_growth_log'].values
        # y_actual, y_median, y_std = y_transform(y_ori, 'encode', func_shift, func_power, dict_transform)
        if dict_transform['aug_size'] > 0:
            x_array_aug = np.concatenate([X_cal for _ in range(dict_transform['aug_size'])], axis=0)
            coeff_array = np.random.randn(x_array_aug.shape[0], x_array_aug.shape[1]) * aug_sigma / dict_transform['std_adjust']
            for feature in [i for i in dict_transform['features_exempt'] if i in features_x]:
                coeff_array[:features_x.index(feature)] = 0
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

    def date_month_convertion(data, bool_end=False):
        """
        Convert between 'yyyy-mm-dd' and number of months
        Args:
            data (str/int): input data either 'yyyy-mm-dd' or number of months
            bool_end (bool): only used in date convertion, whether to output the end of the month

        Returns:
            (str/int): depend on the direction of the convertion return either number of months or 'yyyy-mm-dd'
        """
        if type(data) is int:
            # convert number of months to 'yyyy-mm-dd'
            year = (data) // 12
            month = data % 12
            month_str = str(month + 1).rjust(2, '0')
            if bool_end:
                date_end = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
                output = f'{year}-{month_str}-{date_end[int(month + 1)]}'
            else:
                output = f'{year}-{month_str}-01'

        else:
            # convert 'yyyy-mm-dd' to number of months
            data = str(data)
            output = int(data[:4]) * 12 + int(data[5:7]) - 1
        return output

    def get_holding_value(pd_holding, decision_time_end, bool_keep=False):
        keys = list(pd_holding.keys())
        _pd_holding_record = pd_holding.copy()
        if type(decision_time_end) is str:
            decision_time_end = decision_time_end[:10]
        else:
            decision_time_end = str(decision_time_end)[:10]
        _pd_holding_record['decision_time_end'] = decision_time_end
        _pd_market_cap = stock_price.get_marketcap_time(_pd_holding_record, time_col='decision_time_end')[['symbol', 'marketcap']]
        _pd_holding_record = _pd_holding_record.merge(_pd_market_cap, on='symbol', how='left')
        _pd_holding_record['marketcap'] = _pd_holding_record['marketcap'].fillna(1)
        _pd_holding_record['value'] = _pd_holding_record['shares'] * _pd_holding_record['marketcap']
        if not bool_keep:
            _pd_holding_record = _pd_holding_record[keys + ['value']]
        return _pd_holding_record

    def invest_period_operation(pd_holding, pd_data_operate, dict_decision_time, dict_transform):

        decision_time_final = dict_decision_time['start']
        decision_time_final_end = dict_decision_time['end']
        decision_time_sell_can = str(pd.to_datetime(decision_time_final) - pd.to_timedelta(f'390 days'))[:10]
        decision_time_buy_can = str(pd.to_datetime(decision_time_final) - pd.to_timedelta(f'1800 days'))[:10]
        eval_metric, rate_depreciation = dict_transform['eval_metric'], dict_transform['rate_depreciation']
        rate_step_switch, n_stocks = dict_transform['rate_step_switch'], dict_transform['n_stocks']
        rate_threshold_sell, ratio_threshold_buy = dict_transform['rate_threshold_sell'], dict_transform['ratio_threshold_buy']
        ratio_overhead, margin_interest = dict_transform['ratio_overhead'], dict_transform['margin_interest']

        if 'prepare_data' == 'prepare_data':

            pd_data_train_pre = pd_data_operate.loc[(pd_data_operate.rdq_0 <= decision_time_final) &
                                                    (pd_data_operate['rdq_0'] <= decision_time_final)]

            # prepare the data for the training data
            pd_data_train_list = []
            for tq in (np.arange(4) + 1)[::-1]:
                if tq == 4:
                    pd_temp = pd_data_train_pre.loc[(pd_data_train_pre[f'rdq_pq{tq}'] <= decision_time_final)].copy()
                else:
                    pd_temp = pd_data_train_pre.loc[((pd_data_train_pre[f'rdq_pq{tq + 1}'] >= decision_time_final) |
                                                     pd_data_train_pre[f'rdq_pq{tq + 1}'].isna()) &
                                                    (pd_data_train_pre[f'rdq_pq{tq}'] < decision_time_final)].copy()
                pd_temp['num'] = tq / 4
                pd_temp['marketcap_p'], pd_temp['rdq_p'] = pd_temp[f'marketcap_pq{tq}'], pd_temp[f'rdq_pq{tq}']
                pd_data_train_list.append(pd_temp)
            pd_data_train = pd.concat(pd_data_train_list).sort_values(by=['rdq_0', 'symbol'])
            pd_data_train['datatype'] = 'train'
            head_keys = ['datatype', 'symbol', 'datafqtr', 'num_valid', 'num', 'marketcap_p', 'rdq_p']
            pd_data_train = pd_data_train[head_keys + [i for i in pd_data_train.columns if i not in head_keys]]

            # prepare the data for the stocks candidates to buy
            pd_data_buy_can = pd_data_operate.loc[(pd_data_operate.rdq_0 > decision_time_final) &
                                                  (pd_data_operate.rdq_0 <= decision_time_final_end)].copy()
            pd_data_buy_can['datatype'], pd_data_buy_can['num'] = 'buy', pd_data_buy_can['num_valid']
            pd_data_buy_can['marketcap_b'] = pd_data_buy_can['marketcap_0']
            pd_data_buy_can['rdq_b'] = pd_data_buy_can['rdq_0']

            # prepare the data for the stocks candidates to sell (if any)
            pd_data_sell_can_pre = pd_data_operate.loc[(pd_data_operate['rdq_0'] <= decision_time_final) &
                                                       (pd_data_operate['rdq_0'] >= decision_time_sell_can)]

            pd_data_sell_can_list = []
            for tq in (np.arange(4) + 1)[::-1]:
                pd_temp = pd_data_sell_can_pre.loc[(pd_data_sell_can_pre[f'rdq_pq{tq}'] >= decision_time_final) &
                                                   (pd_data_sell_can_pre[f'rdq_pq{tq}'] <= decision_time_final_end)].copy()
                # num represents the number of quarters from last data available data to current decision time period
                pd_temp['num'] = tq / 4
                pd_temp['marketcap_s'], pd_temp['rdq_s'] = pd_temp[f'marketcap_pq{tq}'], pd_temp[f'rdq_pq{tq}']
                pd_data_sell_can_list.append(pd_temp)
            _pd_data_sell_can = pd.concat(pd_data_sell_can_list).copy().sort_values(by=['rdq_0', 'symbol'])
            _pd_data_sell_can['datatype'] = 'sell'
            head_keys = ['datatype', 'symbol', 'datafqtr', 'num_valid', 'num', 'marketcap_s', 'rdq_s']
            _pd_data_sell_can = _pd_data_sell_can[head_keys + [i for i in _pd_data_sell_can.columns if i not in head_keys]]
            # Get the latest data so that prediction can be more accurate
            pd_filter = _pd_data_sell_can.groupby('symbol').rdq_0.max().reset_index()
            pd_data_sell_can_temp = _pd_data_sell_can.merge(pd_filter, on=['symbol', 'rdq_0'], how='inner')

            # Make sure that the prediction period does NOT extend beyond the num_valid (longest extention of meeting growth standard)
            pd_data_sell_can = pd_data_sell_can_temp.loc[pd_data_sell_can_temp.num <= pd_data_sell_can_temp.num_valid]

            pd_data_eval = pd.concat([pd_data_sell_can, pd_data_buy_can]).copy()
            head_keys = ['datatype', 'symbol', 'datafqtr', 'num_p', 'num_valid', 'num', 'marketcap_b', 'rdq_b']
            pd_data_eval['num_p'] = pd_data_eval['num']
            pd_data_eval = pd_data_eval[head_keys + [i for i in pd_data_eval.columns if i not in head_keys]]

            pd_train = prepage_training_data(pd_data_train)
            pd_train = pd_train.loc[pd_train.num_p >=0.75]

        if 'prediction' == 'prediction':
            if predict_method.lower() == 'sklearn':
                dict_transform, regr_list = get_model_sklearn(pd_train, dict_transform)
            elif predict_method.lower() == 'lstm':
                dict_transform, regr_list = get_model_lstm(pd_train, dict_transform)
            else:
                raise ValueError('predict_method can only be in [lstm, sklearn]')

            log_grow_pred_mean_1, log_grow_pred_median_1, log_grow_pred_std_1 = get_prediction(pd_data_eval, dict_transform, regr_list)
            pd_train_eval = pd_train.loc[pd_train.rdq_0 >= decision_time_buy_can].copy()
            log_grow_pred_mean_2, log_grow_pred_median_2, log_grow_pred_std_2 = get_prediction(pd_train_eval, dict_transform, regr_list)
            if eval_metric == 'log_growth_mc_pred_min':
                pd_data_eval[eval_metric] = log_grow_pred_mean_1 - log_grow_pred_std_1 * 1.5
                pd_train_eval[eval_metric] = log_grow_pred_mean_2 - log_grow_pred_std_2 * 1.5
            elif eval_metric == 'log_growth_mc_pred_mean':
                pd_data_eval[eval_metric] = log_grow_pred_mean_1
                pd_train_eval[eval_metric] = log_grow_pred_mean_2
            elif eval_metric == 'log_growth_mc_pred_median':
                pd_data_eval[eval_metric] = log_grow_pred_median_1
                pd_train_eval[eval_metric] = log_grow_pred_median_2

            head_keys = ['datatype', 'symbol', 'datafqtr', 'num_p', 'marketcap_s', 'rdq_s', eval_metric]
            pd_data_eval = pd_data_eval[head_keys + [i for i in pd_data_eval.columns if i not in head_keys]]
            head_keys = ['datatype', 'symbol', 'datafqtr', 'num_p', eval_metric]
            pd_train_eval = pd_train_eval[head_keys + [i for i in pd_train_eval.columns if i not in head_keys]]

            pd_data_eval_sell = pd_data_eval.loc[pd_data_eval.datatype == 'sell']
            pd_data_eval_list = []
            if len(pd_data_eval_sell) > 0:
                pd_data_eval_sell_filter = pd_data_eval_sell.groupby('symbol').rdq_s.min().reset_index()
                pd_data_eval_sell = pd_data_eval_sell.merge(pd_data_eval_sell_filter, on=['symbol', 'rdq_s'], how='inner')
                pd_data_eval_sell = pd_data_eval_sell.loc[pd_data_eval_sell.symbol.isin(pd_holding.symbol)]
                pd_data_eval_list.append(pd_data_eval_sell)

            pd_data_eval_buy = pd_data_eval.loc[(pd_data_eval.datatype == 'buy') & (pd_data_eval.num_valid == 1)]
            pd_data_eval_list.append(pd_data_eval_buy)

            pd_sell_blind = pd_holding.iloc[1:][['symbol', 'rdq_0']].copy()
            pd_sell_blind['rdq_s'] = (pd_sell_blind['rdq_0'] + pd.to_timedelta('365 day')).astype(str).str[:10]
            pd_sell_blind = pd_sell_blind.loc[pd_sell_blind.rdq_s <= decision_time_final_end]
            pd_sell_blind['datatype'] = 'sell_blind'
            # pd_data_eval_list.append(pd_sell_blind)

            pd_data_eval_operation = pd.concat(pd_data_eval_list)

            pd_data_eval_operation['rdq_operate'] = pd_data_eval_operation['rdq_s']
            _ind = pd_data_eval_operation.datatype == 'buy'
            pd_data_eval_operation.loc[_ind, 'rdq_operate'] = pd_data_eval_operation.loc[_ind, 'rdq_b']
            head_keys = ['datatype', 'symbol', 'datafqtr', 'rdq_operate', 'rdq_0']
            pd_data_eval_operation = pd_data_eval_operation[head_keys + [i for i in pd_data_eval_operation.columns if i not in head_keys]]
            pd_data_eval_operation = pd_data_eval_operation.sort_values(by='rdq_operate')
            pd_data_eval_operation['rdq_0'] = pd.to_datetime(pd_data_eval_operation['rdq_0'])

        def sell_share(pd_holding, _ind, rdq_s=None):
            shares = pd_holding.loc[_ind].iloc[0].shares
            if 'pandas' in str(type(rdq_s)):
                sell_date = rdq_s
            else:
                sell_date = str(rdq_s)[:10]
            pd_temp = pd_holding.loc[_ind].copy()
            pd_temp['rdq_0'] = sell_date
            pd_quote = stock_price.get_marketcap_time(pd_temp, time_col='rdq_0')
            free_cash_gain = shares * pd_quote.iloc[0].marketcap

            pd_holding = pd_holding.loc[~_ind].copy()
            free_cash_current = pd_holding.iloc[0].shares
            pd_holding.iloc[0] = ['free_cash', free_cash_current + free_cash_gain, None, None, None, None, None]
            return pd_holding

        def buy_share(pd_holding, symbol, time_buy):
            a = 1

        rate_depreciation_log = np.log10(1 + rate_depreciation)

        _cut = 0
        # make sure the holding stock is sold after > 390 days after rdq_0
        for i in range(1, len(pd_holding)):
            i = i - _cut
            pd_entry_holding = pd_holding.iloc[i]
            symbol = pd_entry_holding.symbol
            if symbol not in pd_data_eval_operation.symbol:
                rdq_0 = pd_entry_holding.rdq_0
                rdq_0_adjust = str(rdq_0 + pd.to_timedelta('390 day'))[:10]
                rdq_s = pd.to_datetime(pd_entry_holding.rdq_0) + pd.to_timedelta('365 day')
                if rdq_0_adjust < decision_time_final:
                    _ind = (pd_holding.symbol == symbol) & (pd_holding.rdq_0 == rdq_0)
                    pd_holding = sell_share(pd_holding, _ind, rdq_s)
                    _cut += 1
                    print('Sell', len(pd_holding) + 1, symbol, len(pd_holding))

        for i in range(len(pd_data_eval_operation)):
            free_cash = pd_holding.iloc[0]['shares']
            pd_entry = pd_data_eval_operation.iloc[i]
            _operation, _symbol = pd_entry['datatype'], pd_entry['symbol']
            n_holding = len(pd_holding)

            if 'sell' in _operation:
                if _symbol in pd_holding.symbol:
                    _ind = pd_holding.symbol == _symbol
                    _bool_sell = False
                    if str(pd_holding.loc[_ind].iloc[0].rdq_0)[:10] < decision_time_final:
                        # Need to make sure that the time this stock bought was before this decision period
                        if _operation == 'sell_blind':
                            _bool_sell = True
                        elif _operation == 'sell':
                            if pd_entry[eval_metric] / pd_entry['num_p'] < (np.log10(1 + rate_threshold_sell)):
                                _bool_sell = True
                        else:
                            raise ValueError(f"Can't recognize the operation {_operation}")
                    if _bool_sell:
                        pd_holding = sell_share(pd_holding, _ind, pd_entry)
                        # print(f'Sell {_symbol}')
            elif _operation == 'buy':
                eval_metric_value = pd_entry[eval_metric]
                eval_metric_threshold = pd_train_eval.loc[pd_train_eval.num_p == pd_entry.num_p][eval_metric].quantile(1 - ratio_threshold_buy)
                if eval_metric_value >= eval_metric_threshold:
                    if _symbol in list(pd_holding.symbol):
                        # Update the latest prediction result
                        ind_array = pd_holding.symbol == _symbol
                        rdq_0_1st, _shares = pd_holding.loc[ind_array].iloc[0][['rdq_0_1st', 'shares']]
                        pd_holding.loc[ind_array] = [pd_entry.symbol, _shares, rdq_0_1st, pd_entry.rdq_0, pd_entry.rdq_pq4,
                                                     pd_entry[eval_metric], pd_entry.num_p]
                    elif free_cash > 0:
                        # There is free cash, buy anything that's predicted to grow more than depreciation rate
                        rdq_0 = pd_entry.rdq_0
                        n_spot = n_stocks + 1 - len(pd_holding)
                        pd_value_cal = get_holding_value(pd_holding, rdq_0, bool_keep=False)
                        _value_total = pd_value_cal.value.sum()
                        _free_cash_buy_1 = _value_total / n_spot
                        _free_cash_buy_2 = max(free_cash + _value_total * ratio_overhead, _free_cash_buy_1)
                        pd_holding.loc[pd_holding.symbol == 'free_cash', 'shares'] = free_cash - _free_cash_buy_2
                        pd_holding_new = pd.DataFrame({'symbol': [pd_entry.symbol], 'shares': [_free_cash_buy_2 / pd_entry.marketcap_b],
                                                       'rdq_0_1st': [rdq_0],  'rdq_0': [rdq_0], 'rdq_pq4': [pd_entry['rdq_pq4']],
                                                       'pred': [pd_entry[eval_metric]], 'num_p': [pd_entry['num_p']]})
                        pd_holding = pd.concat([pd_holding, pd_holding_new])
                        # print(f'Buy {_symbol}')
                    else:
                        # No free cash, and the symbol is not on the holding list needs to swap stocks
                        # But only swap the stock that was bought on the current period
                        _pd_pred_min = pd_holding.loc[((pd_holding.rdq_0_1st.astype(str) >= decision_time_final) |
                                                       (pd_holding.rdq_0_1st.astype(str) <= decision_time_sell_can)) &
                                                      (pd_holding.symbol != 'free_cash')]
                        if len(_pd_pred_min) > 0:
                            _argmin = np.argmin(_pd_pred_min.pred)
                            _symbol_hold, _pred_min_hold = _pd_pred_min.iloc[_argmin][['symbol', 'pred']]
                            _rdq_0_hold, num_p_hold = _pd_pred_min.iloc[_argmin][['rdq_0_1st', 'num_p']]
                            _depreciation = ((pd_entry.rdq_0 - _rdq_0_hold).days / 365 + pd_entry.num_p - num_p_hold) * rate_depreciation_log
                            _growth_entry = 10 ** pd_entry[eval_metric]
                            _growth_hold = 10 ** _pred_min_hold * (1 + rate_step_switch) * (1 - _depreciation)

                            if _growth_entry > _growth_hold:
                                # There is profit improving opportunity by switching a investment
                                ind_array = pd_holding.symbol == _symbol_hold
                                _pd_sell = pd_holding.loc[ind_array].copy()
                                _pd_sell['rdq_0'] = pd_entry['rdq_0']
                                pd_quote = stock_price.get_marketcap_time(_pd_sell, time_col='rdq_0')
                                _free_cash = pd_quote.iloc[0]['marketcap'] * _pd_sell.iloc[0]['shares']
                                _shares = _free_cash / pd_entry.marketcap_b
                                pd_holding.loc[ind_array] = [pd_entry['symbol'], _shares, pd_entry['rdq_0'], pd_entry['rdq_0'],
                                                             pd_entry['rdq_pq4'], pd_entry[eval_metric], pd_entry['num_p']]
                                #print(f'replace {_symbol_hold} with {pd_entry.symbol} on {str(rdq_0)[:10]}')
                                _operation = 'replace'
            else:
                raise ValueError('operation has to be either buy, sell, or sell_blind.')
            if len(pd_holding) != n_holding:
                print(i, _operation, n_holding, pd_entry.symbol, len(pd_holding))
        return pd_holding

if __name__ == '__main__':
    predict_method = 'sklearn'
    dict_revenue_growth_min = {'1': 0.0, '0': 0.2}
    dict_book_value_growth_min = {'1': 0.0, '0': 0.2}
    dict_revenue_growth_max = {}
    dict_book_value_growth_max = {}
    mc_book_ratio = [2.5, 65]
    mc_revenue_ratio = [2.5, 65]
    evaluate_span_month = 3
    coeff_fade = 0.9
    func_shift, func_power, std_adjust = 3, 3, 2
    features_exempt = ['num', 'num_p']
    eval_metric = 'log_growth_mc_pred_mean'
    rate_depreciation = 0.2
    rate_step_switch = 0
    rate_threshold_sell = -0.5
    ratio_threshold_buy = 0.25
    ratio_overhead = 0.1
    n_stocks = 4
    n_threads = 1
    _decision_time_start, _decision_time_end = '2007-01-01', '2021-12-30'

    #################################################
    # sklearn parameters
    lstm_units = 32
    epochs = 100
    #################################################
    # GB/RF parameters
    regr_type = 'RF'
    n_regr = 10
    aug_size, aug_sigma = 20, 0.1
    n_estimators_min, n_estimators_max = 400, 500
    learning_rate_min, learning_rate_max = 0.85, 1
    max_depth = 5
    booster, subsample = 'gbtree', 0.85
    #################################################

    time_shuffle = 'time'
    marketcap_min, n_year_x = 100, 3
    margin_interest = 0.08
    tree_method, predictor = 'gpu_hist', 'gpu_predictor'
    n_estimators_list = np.linspace(40, 70, n_regr).astype(int)
    learning_rates = np.arange(learning_rate_min, learning_rate_max, (learning_rate_max - learning_rate_min) / n_regr)
    n_estimators_list = list(n_estimators_list) + list(n_estimators_list)
    learning_rates = list(learning_rates) + list(learning_rates)[::-1]
    _ = min(len(n_estimators_list), len(learning_rates))
    n_estimators_list, learning_rate_list = n_estimators_list[:_], learning_rates[:_]


    dict_transform = {'mean': {}, 'std': {}, 'n_year_x': n_year_x, 'func_shift': func_shift, 'func_power': func_power,
                      'aug_size': aug_size, 'aug_sigma': aug_sigma, 'std_adjust': std_adjust, 'lstm_units': lstm_units,
                      'lstm_epochs': epochs, 'coeff_fade': coeff_fade, 'features_exempt': features_exempt,
                      'n_estimators_list': n_estimators_list, 'learning_rates': learning_rates, 'max_depth': max_depth,
                      'tree_method': tree_method, 'predictor': predictor, 'eval_metric': eval_metric,
                      'rate_depreciation': rate_depreciation, 'rate_step_switch': rate_step_switch, 'n_stocks': n_stocks,
                      'rate_threshold_sell': rate_threshold_sell, 'ratio_threshold_buy': ratio_threshold_buy, 'n_threads': n_threads,
                      'regr_type': regr_type, 'booster': booster, 'subsample': subsample, 'ratio_overhead': ratio_overhead,
                      'margin_interest': margin_interest}

    _pd_data = pd_data_ori.copy()
    # Get rid of the data entires should be pre-filtered
    _pd_data = _pd_data.loc[~((_pd_data.marketcap_pq4.isna()) & (_pd_data.rdq_0 < common_func.date(-400)))]
    pd_base = _pd_data
    for i_year in np.arange(4) + 1:
        ind_large = pd_base[f'marketcap_pq{i_year}'] / pd_base[f'marketcap_0'] > 6.7
        ind_small = pd_base[f'marketcap_pq{i_year}'] / pd_base[f'marketcap_0'] < 0.15
        if any(ind_large):
            pd_base.loc[ind_large, f'marketcap_pq{i_year}'] = pd_base.loc[ind_large][f'marketcap_0'] * 6.7
        if any(ind_small):
            pd_base.loc[ind_small, f'marketcap_pq{i_year}'] = pd_base.loc[ind_small][f'marketcap_0'] * 0.15

    for i in dict_revenue_growth_min:
        pd_base = pd_base.loc[(pd_base[f'revenue_{i}'] / pd_base[f'revenue_{int(i) + 1}']) > (1 + dict_revenue_growth_min[i])]
    for i in dict_book_value_growth_min:
        pd_base = pd_base.loc[(pd_base[f'book_value_{i}'] / pd_base[f'book_value_{int(i)+1}']) > (1 + dict_book_value_growth_min[i])]
    # _pd_data = _pd_data.loc[(_pd_data[f'revenue_q0'] / _pd_data[f'revenue_q4']) >= (_pd_data[f'revenue_1'] / _pd_data[f'revenue_2'])]
    # _pd_data = _pd_data.loc[(_pd_data[f'book_value_q0'] / _pd_data[f'book_value_q4']) >=
    #                         (_pd_data[f'book_value_1'] / _pd_data[f'book_value_2'])]
    pd_base = pd_base.loc[(pd_base['revenue_q0'] / pd_base['revenue_q4']) * (pd_base['book_value_q0'] / pd_base['book_value_q4']) >=
                          (pd_base['revenue_1'] / pd_base['revenue_2']) * (pd_base['book_value_1'] / pd_base['book_value_2'])]
    pd_base = pd_base.loc[((pd_base.marketcap_0 / pd_base.book_value_0) <= mc_book_ratio[1]) &
                          ((pd_base.marketcap_0 / pd_base.book_value_0) >= mc_book_ratio[0])]
    pd_base = pd_base.loc[((pd_base.marketcap_0 / pd_base.revenue_0) <= mc_revenue_ratio[1]) &
                          ((pd_base.marketcap_0 / pd_base.revenue_0) >= mc_revenue_ratio[0])]
    pd_base = pd_base.loc[pd_base.marketcap_0 >= marketcap_min]
    pd_base = pd_base.loc[pd_base.status == 'valid'].copy()

    _rank_array_4 = np.asarray(pd_base['rank'])
    dict_rank_array = {i: set(_rank_array_4 - i + 4) for i in (np.arange(4) + 1)}
    _rank_array_pool, pd_data_list = set(), []
    for i in sorted(dict_rank_array, reverse=True):
        if i == 4:
            _rank_array_pool.update(dict_rank_array[i])
        else:
            dict_rank_array[i] = dict_rank_array[i] - _rank_array_pool
            _rank_array_pool = _rank_array_pool.union(set(dict_rank_array[i]))
        pd_data_entry = _pd_data.loc[_pd_data['rank'].isin(dict_rank_array[i])].copy()
        pd_data_entry['num_valid'] = i / 4
        # num_valid represents which growing state is the stock is in,
        # 1 means it meets the standard of full growth last quarter
        # 0.25 means it has not meet the standard for 3 quarters

        if i == 4:
            pd_data_entry['status'] = 'valid'
        else:
            pd_data_entry['status'] = None
        pd_data_list.append(pd_data_entry)
    pd_data_operate = pd.concat(pd_data_list).sort_values(by=['rdq_0', 'symbol']).copy()
    head_keys = ['symbol', 'datafqtr', 'num_valid']
    pd_data_operate = pd_data_operate[head_keys + [i for i in pd_data_operate.columns if i not in head_keys]]

    pd_holding = pd.DataFrame({'symbol': ['free_cash'], 'shares': [10000], 'rdq_0_1st': [None],
                               'rdq_0': [None], 'rdq_pq4': [None], 'pred': [None], 'num_p': [None]})
    pd_holding_fr = None

    _decision_time_start_month = date_month_convertion(_decision_time_start)
    _decision_time_end_month = date_month_convertion(_decision_time_end)
    n_period = (_decision_time_end_month - _decision_time_start_month) // evaluate_span_month

    pd_holding_record_list = []
    time_start = time.time()
    for i_period in range(n_period + 1):
        decision_time_start = date_month_convertion(_decision_time_start_month + i_period * evaluate_span_month, False)
        decision_time_end = date_month_convertion(_decision_time_start_month + (i_period + 1) * evaluate_span_month - 1, True)
        dict_decision_time = {'start': decision_time_start, 'end': decision_time_end}
        pd_holding = invest_period_operation(pd_holding, pd_data_operate, dict_decision_time, dict_transform)

        _pd_holding_record = get_holding_value(pd_holding, decision_time_end, bool_keep=True)
        value_total = int(_pd_holding_record.value.sum())

        pd_holding_record_list.append(_pd_holding_record)
        time_span = round(time.time() - time_start, 1)
        print(f'{time_span} s - completed investing in {decision_time_start} - {i_period + 1}/{n_period} - value {value_total}')

    pd_holding_record = pd.concat(pd_holding_record_list)

    comp_growth_rate = round((10 ** (np.log10(value_total / 10000) / (len(pd_holding_record_list)/4)) - 1) * 100, 2)
    print(f'Final compounded annual growth rate {comp_growth_rate}%')