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
                           0: {0.051: ['Revenue - Total'],
                               0.05: ['Stockholders Equity - Total']},
                           1: {0.035: ['Revenue - Total']}, }

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
    for key_time in range(n_year_x + 1):
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


    merge_query = ''
    for key_time in range(n_year_x + 1):
        for key_item in col_output_list_avg:
            merge_query += f', ty{key_time}.{key_item}_{key_time}'

    query_translate = ' rank, symbol, datafqtr, '
    for key_time in range(n_year_x + 1):
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
        where num = 16
        order by symbol, rdq, datafqtr
    ),
    data0 as (
        select tf.rank, tf.symbol, tf.datafqtr, tf.rdq, {col_query_avg_0}
        from filter_4_year tf, filter_1 ty0
        where tf.symbol = ty0.symbol 
        and tf.rank - ty0.rank >= 0 
        and tf.rank - ty0.rank <= 3 
        group by tf.symbol, tf.rdq, tf.datafqtr
        order by tf.symbol, tf.rdq, tf.datafqtr
    ),
    data1 as (
        select tf.rank, tf.symbol, tf.rdq, {col_query_avg_1}
        from filter_4_year tf, filter_1 ty1
        where tf.symbol = ty1.symbol
        and tf.rank - ty1.rank >= 4
        and tf.rank - ty1.rank <= 7
        group by tf.symbol, tf.rdq, tf.datafqtr
        order by tf.symbol, tf.rdq, tf.datafqtr
    ),
    data2 as (
        select tf.rank, tf.symbol, tf.rdq, {col_query_avg_2}
        from filter_4_year tf, filter_1 ty2
        where tf.symbol = ty2.symbol 
        and tf.rank - ty2.rank >=  8 
        and tf.rank - ty2.rank <= 11
        group by tf.symbol, tf.rdq, tf.datafqtr
        order by tf.symbol, tf.rdq, tf.datafqtr
    ),
    data3 as (
        select tf.rank, tf.symbol, tf.rdq, {col_query_avg_3}
        from filter_4_year tf, filter_1 ty3
        where tf.symbol = ty3.symbol
        and tf.rank - ty3.rank >=  12
        and tf.rank - ty3.rank >=  15
        group by tf.symbol, tf.rdq, tf.datafqtr
        order by tf.symbol, tf.rdq, tf.datafqtr
    ),
    
    data_all as (
        select ty0.rank, ty0.symbol, ty0.datafqtr {merge_query}
        from data0 ty0 inner join data1 ty1 inner join data2 ty2 inner join data3 ty3
        on ty0.symbol = ty1.symbol and ty0.symbol = ty2.symbol and ty0.symbol = ty3.symbol
        and ty0.rdq = ty1.rdq and ty0.rdq = ty2.rdq and ty0.rdq = ty3.rdq
    ), 
    
    data_translate as (
        select {query_translate} from data_all
    )
    
    select * from data_translate
    {query_growth_filter}

    """
    # print(command_query)
    pd_data_raw = pd.read_sql(command_query, con)
    pd_data_raw = pd_data_raw[[i for i in pd_data_raw.keys() if i != 'rank']]

    # Add marketcap info
    pd_marketcap_report_0 = stock_price.get_marketcap_time(pd_data_raw, time_col='rdq_0')
    pd_marketcap_report_1 = stock_price.get_marketcap_time(pd_data_raw, time_col='rdq_1')

    pd_marketcap_report_0 = pd_marketcap_report_0.rename(columns={'marketcap': 'marketcap_0'})
    pd_marketcap_report_1 = pd_marketcap_report_1.rename(columns={'marketcap': 'marketcap_1'})

    pd_data = pd_data_raw.merge(pd_marketcap_report_0[['symbol', 'rdq_0', 'marketcap_0']], on=['symbol', 'rdq_0'], how='inner')
    pd_data = pd_data.merge(pd_marketcap_report_1[['symbol', 'rdq_1', 'marketcap_1']], on=['symbol', 'rdq_1'], how='inner')
    keys_front = ['symbol', 'datafqtr', 'marketcap_0', 'marketcap_1']
    pd_data = pd_data[keys_front + [i for i in pd_data.columns if i not in keys_front]]
    pd_data = pd_data.sort_values(by='marketcap_0')
    pd_data_ori = pd_data.copy()


def plot_dist(pd_mdata, features_x):
    fig, ax = plt.subplots(4, 5, figsize=(15, 7.5))
    ax = fig.axes
    for i, feature in enumerate(features_x):
        dist = pd_mdata[feature]
        ax[i].hist(dist, bins=35)
        ax[i].set_title(feature)
    fig.tight_layout()


def get_model(pd_data, n_year_x):
    dict_transform = {'mean': {}, 'std': {}, 'n_year_x': n_year_x}
    pd_mdata = (pd_data['marketcap_0'] / pd_data['marketcap_1']).rename('mc_growth').reset_index()[['mc_growth']]
    pd_mdata['mc_growth_log'] = np.log10(pd_mdata['mc_growth'])
    pd_mdata['mc_growth_log_squred'] = pd_mdata['mc_growth_log'] ** 2

    features_bvr = ['cur_asset', 'cur_liab', 'cash_invest', 'cash_flow', 'revenue', 'profit']
    features_x = ['mc_bv']
    pd_mdata['mc_bv'] = pd_data[f'marketcap_1'] / pd_data[f'book_value_1']
    for i_year in range(1, n_year_x):
        feature_x = f'bv_growth_{i_year}'
        pd_mdata[feature_x] = pd_data[f'book_value_{i_year}'] / pd_data[f'book_value_{i_year + 1}']
        features_x.append(feature_x)

    for i_year in range(1, n_year_x + 1):
        for feature in features_bvr:
            feature_x = f'bvr_{feature}_{i_year}'
            pd_mdata[feature_x] = pd_data[f'{feature}_{i_year}'] / pd_data[f'book_value_{i_year}']
            features_x.append(feature_x)

    for feature in features_x:
        col = np.log10(pd_mdata[feature])
        mean, std = col.mean(), col.std()
        dict_transform['mean'][feature] = mean
        dict_transform['std'][feature] = std
        col = (col - mean) / std
        pd_mdata[feature] = col

    pd_mdata_cal = pd_mdata
    # regr = RandomForestRegressor(max_depth=3, n_estimators=2500)
    regr = GradientBoostingRegressor(max_depth=3, n_estimators=250)
    X_train, y_train = pd_mdata_cal[features_x].values, pd_mdata_cal['mc_growth_log'].values
    regr.fit(X_train, y_train)
    dict_transform['features_x'] = features_x
    return dict_transform, regr


def get_prediction(pd_data, dict_transform, regr):
    n_year_x = dict_transform['n_year_x']
    pd_mdata = (pd_data['marketcap_0'] / pd_data['marketcap_1']).rename('mc_growth').reset_index()[['mc_growth']]
    pd_mdata['mc_growth_log'] = np.log10(pd_mdata['mc_growth'])
    pd_mdata['mc_growth_log_squred'] = pd_mdata['mc_growth_log'] ** 2

    features_bvr = ['cur_asset', 'cur_liab', 'cash_invest', 'cash_flow', 'revenue', 'profit']
    pd_mdata['mc_bv'] = list(pd_data[f'marketcap_1'] / pd_data[f'book_value_1'])
    for i_year in range(1, n_year_x):
        feature_x = f'bv_growth_{i_year}'
        pd_mdata[feature_x] = list(pd_data[f'book_value_{i_year}'] / pd_data[f'book_value_{i_year + 1}'])


    for i_year in range(1, n_year_x + 1):
        for feature in features_bvr:
            feature_x = f'bvr_{feature}_{i_year}'
            pd_mdata[feature_x] = list(pd_data[f'{feature}_{i_year}'] / pd_data[f'book_value_{i_year}'])

    for feature in dict_transform['mean']:
        col = np.log10(pd_mdata[feature])
        mean, std = dict_transform['mean'][feature], dict_transform['std'][feature]
        col = (col - mean) / std
        pd_mdata[feature] = col

    X_cal = pd_mdata[dict_transform['features_x']].values
    y_pred = regr.predict(X_cal)
    return y_pred




if 1 == 1:
    mc_book_ratio = [1, 75]
    marketcap_min = 100
    ratio_train = 0.8
    n_year_x = 3

    pd_data = pd_data_ori.copy()
    pd_data = pd_data.loc[(pd_data.marketcap_0 / pd_data.marketcap_1) <= 75]

    pd_data = pd_data.loc[(pd_data.marketcap_1 / pd_data.book_value_1 <= mc_book_ratio[1]) &
                          (pd_data.marketcap_1 / pd_data.book_value_1 >= mc_book_ratio[0])]
    pd_data = pd_data.loc[pd_data.marketcap_1 >= marketcap_min]
    pd_data = pd_data.sort_values(by='rdq_0')
    pd_data.index = np.arange(len(pd_data))

    n_threshold = pd_data.iloc[int(len(pd_data) * ratio_train)].name
    pd_data_train, pd_data_test = pd_data.iloc[:n_threshold].copy(), pd_data.iloc[n_threshold:].copy()
    dict_transform, regr = get_model(pd_data_train, n_year_x)
    log_grow_pred_train = get_prediction(pd_data_train, dict_transform, regr)
    log_grow_pred_test = get_prediction(pd_data_test, dict_transform, regr)

    head_keys = ['symbol', 'datafqtr', 'marketcap_0', 'marketcap_1', 'log_growth_mc', 'log_growth_mc_pred']
    pd_data_train['log_growth_mc_pred'] = log_grow_pred_train
    pd_data_test['log_growth_mc_pred'] = log_grow_pred_test
    pd_data_train['log_growth_mc'] = np.log10(pd_data_train['marketcap_0'] / pd_data_train['marketcap_1'])
    pd_data_test['log_growth_mc'] = np.log10(pd_data_test['marketcap_0'] / pd_data_test['marketcap_1'])
    pd_data_train = pd_data_train[head_keys + [i for i in pd_data_train.columns if i not in head_keys]]
    pd_data_test = pd_data_test[head_keys + [i for i in pd_data_test.columns if i not in head_keys]]
    pd_data_test_best = pd_data_test.loc[pd_data_test.rdq_0 >= '2020-09-31'].sort_values(by='log_growth_mc_pred')

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(pd_data_train['log_growth_mc_pred'], pd_data_train['log_growth_mc'], '.')
    ax[0].set_xlabel('Predicted Growth')
    ax[0].set_ylabel('Actual Growth')
    ax[0].set_title('Training set')
    ax[1].plot(pd_data_test['log_growth_mc_pred'], pd_data_test['log_growth_mc'], '.')
    ax[1].set_xlabel('Predicted Growth')
    ax[1].set_ylabel('Actual Growth')
    ax[1].set_title('Testing set')

    pd_regr_train = pd.DataFrame({'y': pd_data_train['log_growth_mc'], 'y_pred': pd_data_train['log_growth_mc_pred']})
    pd_regr_test = pd.DataFrame({'y': pd_data_test['log_growth_mc'], 'y_pred': pd_data_test['log_growth_mc_pred']})


    thresholds = [0.2, 0.25, 0.3]
    n_fig = len(thresholds)
    fig, ax = plt.subplots(1, n_fig, figsize=(n_fig * 4.5, 4.5))
    for i, threshold in enumerate(thresholds):
        if i < (len(thresholds) - 1):
            pd_select = pd_regr_test.loc[(pd_regr_test.y_pred >= thresholds[i]) & (pd_regr_test.y_pred < thresholds[i+1])]
        else:
            pd_select = pd_regr_test.loc[(pd_regr_test.y_pred >= thresholds[i])]
        ax[i].hist(pd_select.y, bins=20)
    fig.tight_layout()

if 1 == 0:
    pd_data = pd_data_ori.copy()
    key_head_list = ['rank', 'symbol', 'rdq', 'datafqtr', 'quarter']
    cols = key_head_list + sorted([i for i in list(pd_data.columns) if i not in key_head_list])
    pd_data['quarter'] = pd_data['rdq'].str[:4].astype(int) + ((pd_data['rdq'].str[5:7].astype(int) - 1) // 3) / 4

    pd_data = pd_data[cols]

    pd_price = stock_price.get_price_pd_query(pd_data[['symbol', 'rdq']])
    pd_price_latest = stock_price.get_price_latest(list(pd_data['symbol']))

    pd_select = pd_data.loc[(pd_data.quarter >= 2000) & (pd_data.quarter <= 2003)]
    # pd_data['rdq'] = pd.to_datetime(pd_data['rdq'])

    pd_size = pd_data.groupby('quarter').size().rename('num').reset_index()
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax = fig.axes
    ax[0].plot(pd_size.quarter, pd_size.num, '.-')

if 1 == 0:
    # stock_price = StockPrice()
    pd_data = pd_data_ori.copy()
    pd_price = stock_price.get_price_pd_query(pd_data_ori)

    pd_price = pd_price.rename(columns={'time': 'time_price', 'time_request': 'rdq', 'close': 'price'})
    pd_data = pd_data.merge(pd_price, on=['symbol', 'rdq'], how='inner')
    pd_data['market_cap'] = pd_data['shares'] * pd_data['price']

    key_head_list = ['rank', 'symbol', 'rdq', 'datafqtr', 'time_price', 'price', 'market_cap', 'shares']
    cols = key_head_list + sorted([i for i in list(pd_data.columns) if i not in key_head_list])
    pd_data = pd_data[cols]

# con.close()