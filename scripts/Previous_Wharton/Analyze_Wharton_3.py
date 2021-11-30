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
import xgboost

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
                           0: {0.015: ['Revenue - Total'],
                               0.0151: ['Stockholders Equity - Total']},
                           1: {0.005: ['Revenue - Total']}, }

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
    year_list_1 = list(range(n_year_x))
    for key_time in year_list_1:
        for col in year_col_positive_list:
            query_growth_filter += f' and {col}_{key_time} > 0'

    col_query = ', '.join([i for i in col_output_list if (i in dict_col_name.values()) & (i != 'rdq')])
    for key in desc_new_dict:
        key1 = dict_col_name[desc_new_dict[key][0]]
        key2 = dict_col_name[desc_new_dict[key][1]]
        col_query += f', {key1} - {key2} as {key}'
    col_output_list_avg = [i for i in col_output_list if i != 'cshoq']
    col_query_avg_0 = 'max(ty0.rdq) as rdq_0, ' + ', '.join([f'avg(ty0.{i}) as {i}_0' for i in col_output_list_avg if i != 'rdq'])
    col_query_avg_1 = 'max(ty1.rdq) as rdq_1, ' + ', '.join([f'avg(ty1.{i}) as {i}_1' for i in col_output_list_avg if i != 'rdq'])
    col_query_avg_2 = 'max(ty2.rdq) as rdq_2, ' + ', '.join([f'avg(ty2.{i}) as {i}_2' for i in col_output_list_avg if i != 'rdq'])
    col_query_avg_p = 'max(typ.rdq) as rdq_p, ' + ', '.join([f'avg(typ.{i}) as {i}_p' for i in col_output_list_avg if i != 'rdq'])
    col_query_avg_q0 = 'max(tq.rdq) as rdq_q0, ' + ', '.join([f'avg(tq.{i}) as {i}_q0' for i in col_output_list_avg if i != 'rdq'])
    col_query_avg_q1 = 'max(tq.rdq) as rdq_q1, ' + ', '.join([f'avg(tq.{i}) as {i}_q1' for i in col_output_list_avg if i != 'rdq'])
    col_query_avg_q4 = 'max(tq.rdq) as rdq_q4, ' + ', '.join([f'avg(tq.{i}) as {i}_q4' for i in col_output_list_avg if i != 'rdq'])

    merge_query_x = ''
    year_list = list(range(n_year_x))
    for key_time in year_list:
        for key_item in col_output_list_avg:
            merge_query_x += f', ty{key_time}.{key_item}_{key_time}'
    quarter_list = [0, 1, 4]
    for key_time in quarter_list:
        for key_item in col_output_list_avg:
            merge_query_x += f', tq{key_time}.{key_item}_q{key_time}'

    merge_query_y = ''
    for key_time in ['p']:
        for key_item in col_output_list_avg:
            merge_query_y += f', ty{key_time}.{key_item}_{key_time}'

    query_translate = ' rank, symbol, datafqtr, num_p, '
    for key_time in year_list + ['p']:
        for col in col_output_list_avg:
            if col in dict_col_name_reverse:
                query_translate += f'{col}_{key_time} as {desc_output_dict[dict_col_name_reverse[col]]}_{key_time}, '
            else:
                query_translate += f'{col}_{key_time} as {desc_output_dict[col]}_{key_time}, '
        query_translate += '\n'
    for key_time in quarter_list:
        for col in col_output_list_avg:
            if col in dict_col_name_reverse:
                query_translate += f'{col}_q{key_time} as {desc_output_dict[dict_col_name_reverse[col]]}_q{key_time}, '
            else:
                query_translate += f'{col}_q{key_time} as {desc_output_dict[col]}_q{key_time}, '
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
    data0 as (
        select tf.rank, tf.symbol, tf.datafqtr, tf.rdq, {col_query_avg_0}
        from filter_3_year tf, filter_1 ty0
        where tf.symbol = ty0.symbol 
        and tf.rank - ty0.rank >= 0 
        and tf.rank - ty0.rank <= 3 
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    data1 as (
        select tf.rank, tf.symbol, tf.rdq, {col_query_avg_1}
        from filter_3_year tf, filter_1 ty1
        where tf.symbol = ty1.symbol
        and tf.rank - ty1.rank >= 4
        and tf.rank - ty1.rank <= 7
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    data2 as (
        select tf.rank, tf.symbol, tf.rdq, {col_query_avg_2}
        from filter_3_year tf, filter_1 ty2
        where tf.symbol = ty2.symbol 
        and tf.rank - ty2.rank >=  8 
        and tf.rank - ty2.rank <= 11
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    dataq0 as (
        select tf.rank, tf.symbol, tf.rdq, {col_query_avg_q0}
        from filter_3_year tf, filter_1 tq
        where tf.symbol = tq.symbol and tf.rank = tq.rank 
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    dataq1 as (
        select tf.rank, tf.symbol, tf.rdq, {col_query_avg_q1}
        from filter_3_year tf, filter_1 tq
        where tf.symbol = tq.symbol and tf.rank - tq.rank = 1 
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    dataq4 as (
        select tf.rank, tf.symbol, tf.rdq, {col_query_avg_q4}
        from filter_3_year tf, filter_1 tq
        where tf.symbol = tq.symbol and tf.rank - tq.rank = 4 
        group by tf.symbol, tf.rdq, tf.datafqtr
    ),
    datap as (
        with data_temp as (
            select tf.symbol, tf.rdq, count(typ.rdq) as num_p, {col_query_avg_p}
            from filter_3_year tf, filter_0 typ
            where tf.symbol = typ.symbol 
            and typ.rank - tf.rank >= 1 
            and typ.rank - tf.rank <= 4
            group by tf.symbol, tf.rdq, tf.datafqtr
            order by tf.symbol, tf.rdq, tf.datafqtr
        )
        select * from data_temp
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
    data_merge_xy as (
        select tyx.*, typ.num_p {merge_query_y}
        from data_merge_x tyx left join datap typ
        on tyx.symbol = typ.symbol and tyx.rdq = typ.rdq 
    ), 
    data_translate as (
        select {query_translate} from data_merge_xy
    )

    select * from data_translate
    {query_growth_filter}

    """
    # print(command_query)
    pd_data_raw = pd.read_sql(command_query, con)
    pd_data_raw = pd_data_raw[[i for i in pd_data_raw.keys() if i != 'rank']]
    print('Completed Wharton financial report data pull')
    # Add marketcap info
    pd_marketcap_report_0 = stock_price.get_marketcap_time(pd_data_raw, time_col='rdq_0', avg=14)
    pd_marketcap_report_1 = stock_price.get_marketcap_time(pd_data_raw, time_col='rdq_1', avg=14)
    pd_marketcap_report_q1 = stock_price.get_marketcap_time(pd_data_raw, time_col='rdq_q1', avg=14)
    pd_marketcap_report_p = stock_price.get_marketcap_time(pd_data_raw, time_col='rdq_p', avg=14)

    print('Completed market report data pull')
    pd_marketcap_report_0 = pd_marketcap_report_0.rename(columns={'marketcap': 'marketcap_0'})
    pd_marketcap_report_1 = pd_marketcap_report_1.rename(columns={'marketcap': 'marketcap_1'})
    pd_marketcap_report_q1 = pd_marketcap_report_q1.rename(columns={'marketcap': 'marketcap_q1'})
    pd_marketcap_report_p = pd_marketcap_report_p.rename(columns={'marketcap': 'marketcap_p'})

    pd_data = pd_data_raw.merge(pd_marketcap_report_0[['symbol', 'rdq_0', 'marketcap_0']], on=['symbol', 'rdq_0'], how='inner')
    pd_data = pd_data.merge(pd_marketcap_report_1[['symbol', 'rdq_1', 'marketcap_1']], on=['symbol', 'rdq_1'], how='inner')
    pd_data = pd_data.merge(pd_marketcap_report_q1[['symbol', 'rdq_q1', 'marketcap_q1']], on=['symbol', 'rdq_q1'], how='inner')
    pd_data = pd_data.merge(pd_marketcap_report_p[['symbol', 'rdq_p', 'marketcap_p']], on=['symbol', 'rdq_p'], how='left')
    keys_front = ['symbol', 'datafqtr', 'marketcap_p', 'marketcap_0', 'marketcap_1', 'marketcap_q1']
    pd_data = pd_data[keys_front + [i for i in pd_data.columns if i not in keys_front]]
    pd_data = pd_data.sort_values(by='marketcap_0')
    pd_data_ori = pd_data.copy()
    print('Completed data aggregation')

if 'Define Function' == 'Define Function':
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


    def y_transform(y, direction='encode', func_shift=1, func_power=2):
        if direction.lower() == 'encode':
            y_output = (y + func_shift) ** func_power
        elif direction.lower() == 'decode':
            y_output = (y ** (1 / func_power)) - func_shift
        else:
            raise ValueError(f'direction can only be either encode or decode, input is {direction}')
        return y_output


    def prepare_features(pd_data):
        pd_mdata = (pd_data['marketcap_p'] / pd_data['marketcap_0']).rename('mc_growth').reset_index()[['mc_growth']]
        pd_mdata['mc_growth_log'] = list(np.log10(pd_mdata['mc_growth']))
        pd_mdata['mc_growth_log_squred'] = list(pd_mdata['mc_growth_log'] ** 2)

        features_bvr_year = ['cur_asset', 'cur_liab', 'cash_invest', 'cash_flow', 'revenue', 'profit']
        features_growth = ['book_value', 'revenue']
        mc_bv_years = 2
        for mc_bv_year in range(mc_bv_years):
            pd_mdata[f'mc_bv_{mc_bv_year}'] = list(pd_data[f'marketcap_{mc_bv_year}'] / pd_data[f'book_value_{mc_bv_year}'])
        pd_mdata['mc_bv_q1'] = list(pd_data[f'marketcap_q1'] / pd_data[f'book_value_q1'])

        features_x = [f'mc_bv_{mc_bv_year}' for mc_bv_year in range(mc_bv_years)] + ['mc_bv_q1']

        for feature in features_growth:
            for i_year in range(n_year_x - 1):
                feature_x = f'{feature}_growth_{i_year}'
                pd_mdata[feature_x] = list(pd_data[f'{feature}_{i_year}'] / pd_data[f'{feature}_{i_year + 1}'])
                features_x.append(feature_x)

            for quarter in [1, 4]:
                feature_x = f'{feature}_growth_q{quarter}'
                pd_mdata[feature_x] = list(pd_data[f'{feature}_q0'] / pd_data[f'{feature}_q{quarter}'])
                features_x.append(feature_x)

        for i_year in range(n_year_x):
            for feature in features_bvr_year:
                feature_x = f'bvr_{feature}_{i_year}'
                pd_mdata[feature_x] = list(pd_data[f'{feature}_{i_year}'] / pd_data[f'book_value_{i_year}'])
                features_x.append(feature_x)
        return pd_mdata, features_x


    def get_model(pd_data, dict_transform, n_estimators=500, learning_rate=1, max_depth=3, tree_method=None,
                  predictor=None):

        n_estimators_list = n_estimators if type(n_estimators) in [list, np.ndarray, range] else [n_estimators]
        learning_rate_list = learning_rate if type(learning_rate) in [list, np.ndarray, range] else [learning_rate]

        n_year_x = dict_transform['n_year_x']
        func_shift, func_power = dict_transform['func_shift'], dict_transform['func_power']
        aug_size, aug_sigma = dict_transform['aug_size'], dict_transform['aug_sigma']
        pd_mdata, features_x = prepare_features(pd_data)

        for feature in features_x:
            col = np.log10(pd_mdata[feature])
            mean, std = col.mean(), col.std()
            dict_transform['mean'][feature] = mean
            dict_transform['std'][feature] = std
            col = (col - mean) / std
            pd_mdata[feature] = col

        pd_mdata_cal = pd_mdata
        n_extra = aug_size * len(pd_mdata_cal)
        pd_mdata_cal_aug = pd.concat([pd_mdata_cal for _ in range(int(np.ceil(aug_size)))]).iloc[:n_extra].copy()
        for feature in features_x:
            coeff = np.random.randn(len(pd_mdata_cal_aug)) * aug_sigma
            pd_mdata_cal_aug[feature] = pd_mdata_cal_aug[feature] + coeff
        pd_mdata_cal = pd.concat([pd_mdata_cal, pd_mdata_cal_aug])
        # regr = RandomForestRegressor(max_depth=3, n_estimators=2500)
        # regr = GradientBoostingRegressor(max_depth=3, n_estimators=n_estimators)
        regr_list = []
        x_train, y_train = pd_mdata_cal[features_x].values, pd_mdata_cal['mc_growth_log'].values
        y_train = y_transform(y_train, 'encode', func_shift, func_power)
        time_start = time.time()
        for i_regr in range(len(n_estimators_list)):
            n_estimators = n_estimators_list[i_regr]
            learning_rate = learning_rate_list[i_regr]
            regr = xgboost.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                        predictor=predictor, tree_method=tree_method, random_state=np.random.randint(999999))
            regr.fit(x_train, y_train)
            regr_list.append(regr)
            time_span = round(time.time() - time_start, 1)
            print(f'\rCompleted regression {i_regr + 1}/{len(n_estimators_list)} - Time {time_span} s', end='')
        dict_transform['features_x'] = features_x
        return dict_transform, regr_list


    def get_prediction(pd_data, dict_transform, regr_list, center_type='median'):
        if type(regr_list) is list:
            regr_list = regr_list
        else:
            regr_list = [regr_list]
        # n_year_x = dict_transform['n_year_x']
        # features_x = dict_transform['features_x']
        func_shift, func_power = dict_transform['func_shift'], dict_transform['func_power']
        pd_mdata, features_x = prepare_features(pd_data)

        for feature in dict_transform['mean']:
            col = np.log10(pd_mdata[feature])
            mean, std = dict_transform['mean'][feature], dict_transform['std'][feature]
            col = (col - mean) / std
            pd_mdata[feature] = col

        X_cal = pd_mdata[features_x].values
        y_pred_list = []
        for i_regr, regr in enumerate(regr_list):
            y_pred = y_transform(regr.predict(X_cal), 'decode', func_shift, func_power).reshape(X_cal.shape[0], 1)
            y_pred_list.append(y_pred)
            for _ in range(dict_transform['aug_size']):
                # for _ in range(1):
                pd_mdata_aug = pd_mdata[features_x].copy()
                for feature in features_x:
                    coeff = np.random.randn(len(pd_mdata_aug)) * aug_sigma
                    pd_mdata_aug[feature] = pd_mdata_aug[feature] + coeff
                X_cal = pd_mdata_aug[features_x].values
                y_pred_aug = y_transform(regr.predict(X_cal), 'decode', func_shift, func_power).reshape(X_cal.shape[0], 1)
                y_pred_list.append(y_pred_aug)
        y_pred_concat = np.concatenate(y_pred_list, axis=1)
        if center_type.lower() == 'mean':
            y_pred_mean = y_pred_concat.mean(axis=1)
        elif center_type.lower() == 'median':
            y_pred_mean = np.quantile(y_pred_concat, 0.5, axis=1)
        else:
            raise ValueError('center_type has to be in ["mean", "median"]')
        y_pred_std = y_pred_concat.std(axis=1)
        return y_pred_mean, y_pred_std


    def e2e_pred_data(pd_data, dict_transform, regr_list):
        log_grow_pred_mean, log_grow_pred_std = get_prediction(pd_data, dict_transform, regr_list)

        head_keys = ['symbol', 'datafqtr', 'marketcap_p', 'marketcap_0', 'log_growth_mc', 'log_growth_mc_pred_min',
                     'log_growth_mc_pred_mean', 'log_growth_mc_pred_std']
        pd_data['log_growth_mc_pred_mean'] = log_grow_pred_mean
        pd_data['log_growth_mc_pred_std'] = log_grow_pred_std
        pd_data['log_growth_mc_pred_min'] = log_grow_pred_mean - log_grow_pred_std * 2

        pd_data['log_growth_mc'] = np.log10(pd_data['marketcap_p'] / pd_data['marketcap_0'])
        pd_data = pd_data[head_keys + [i for i in pd_data.columns if i not in head_keys]]
        return pd_data


if 'Training' == 'Training':
    mc_book_ratio = [1, 65]
    marketcap_min = 100
    ratio_train = 0.65
    ratio_dev = 0.25
    n_year_x = 3
    func_shift, func_power = 2, 2
    n_regr = 5
    n_estimators_min, n_estimators_max = 750, 950
    learning_rate_min, learning_rate_max = 0.85, 1
    max_depth, tree_method, predictor = 3, 'gpu_hist', 'gpu_predictor'
    n_estimators_list = range(n_estimators_min, n_estimators_max, (n_estimators_max - n_estimators_min) // n_regr)
    learning_rate_list = np.arange(learning_rate_min, learning_rate_max, (learning_rate_max - learning_rate_min) / n_regr)
    n_estimators_list = list(n_estimators_list) + list(n_estimators_list)
    learning_rate_list = list(learning_rate_list) + list(learning_rate_list)[::-1]
    _ = min(len(n_estimators_list), len(learning_rate_list))
    n_estimators_list, learning_rate_list = n_estimators_list[:_], learning_rate_list[:_]

    aug_size, aug_sigma = 15, 0.2

    pd_data = pd_data_ori.loc[pd_data_ori.num_p >= 4].copy()
    pd_data_extra = pd_data_ori.loc[(pd_data_ori.num_p < 4) | (pd_data_ori.num_p.isna())].copy()

    pd_data = pd_data.loc[((pd_data.marketcap_p / pd_data.marketcap_0) <= 6) &
                          ((pd_data.marketcap_p / pd_data.marketcap_0) >= 10 ** -1)]
    pd_data['datafqtr'] = pd_data['rdq_0'].str[:4].astype(float) + (pd_data['rdq_0'].str[5:7].astype(float) + 1) / 12
    pd_data = pd_data.loc[((pd_data.marketcap_0 / pd_data.book_value_0) <= mc_book_ratio[1]) &
                          ((pd_data.marketcap_0 / pd_data.book_value_0) >= mc_book_ratio[0])]
    pd_data = pd_data.loc[pd_data.marketcap_0 >= marketcap_min]

    pd_data_extra = pd_data_extra.loc[((pd_data_extra.marketcap_p / pd_data_extra.marketcap_0) <= 6) &
                                      ((pd_data_extra.marketcap_p / pd_data_extra.marketcap_0) >= 10 ** -1)]
    pd_data_extra = pd_data_extra.loc[((pd_data_extra.marketcap_0 / pd_data_extra.book_value_0) <= mc_book_ratio[1]) &
                                      ((pd_data_extra.marketcap_0 / pd_data_extra.book_value_0) >= mc_book_ratio[0])]
    pd_data_extra = pd_data_extra.loc[pd_data_extra.marketcap_0 >= marketcap_min]
    pd_data = pd_data.sort_values(by='rdq_0')
    pd_data.index = np.arange(len(pd_data))
    n_threshold_train, n_threshold_dev = int(len(pd_data) * ratio_train), int(len(pd_data) * ratio_dev)
    pd_data_train = pd_data.iloc[:n_threshold_train].copy()
    pd_data_dev = pd_data.iloc[n_threshold_train:(n_threshold_train + n_threshold_dev)].copy()
    pd_data_test = pd.concat([pd_data.iloc[(n_threshold_train + n_threshold_dev):],
                              pd_data_extra.loc[(pd_data_extra.num_p >= 3)]]).copy()
    pd_data_test_extra = pd_data_extra.loc[(pd_data_extra.num_p < 3)].copy()

    regr_list_global = []
    dict_transform = {'mean': {}, 'std': {}, 'n_year_x': n_year_x, 'func_shift': func_shift, 'func_power': func_power,
                      'aug_size': aug_size, 'aug_sigma': aug_sigma}
    dict_transform, regr_list_global = get_model(pd_data_train, dict_transform, n_estimators=n_estimators_list,
                                                 learning_rate=learning_rate_list, max_depth=max_depth, tree_method=tree_method,
                                                 predictor=predictor)

    pd_data_train = e2e_pred_data(pd_data_train, dict_transform, regr_list_global)
    pd_data_dev = e2e_pred_data(pd_data_dev, dict_transform, regr_list_global)
    pd_data_test = e2e_pred_data(pd_data_test, dict_transform, regr_list_global)
    pd_data_test_extra = e2e_pred_data(pd_data_test_extra, dict_transform, regr_list_global)

    if 1 == 0:
        pd_data_train_plot = pd_data_train.copy()
        pd_data_dev_plot = pd_data_dev.copy()
        pd_data_test_plot = pd_data_test.copy()
        pd_data_test_extra_plot = pd_data_test_extra.copy()
        pd_data_dev_plot = pd_data_dev_plot.loc[pd_data_dev_plot['log_growth_mc_pred_min'] >= 0.03]
        pd_data_test_plot = pd_data_test_plot.loc[pd_data_test_plot['log_growth_mc_pred_min'] >= 0.03]
        pd_data_test_extra_plot = pd_data_test_extra_plot.loc[pd_data_test_extra_plot['log_growth_mc_pred_min'] >= 0.03]

    metric_plot = 'log_growth_mc_pred_min'
    # metric_plot = 'log_growth_mc_pred_mean'

    if 1 == 1:
        val_min_threshold = 0.0
        fig, ax = plt.subplots(2, 4, figsize=(18, 9))
        for i_plot in range(2):
            val_min = val_min_threshold if i_plot == 0 else -5
            pd_data_train_plot = pd_data_train.loc[pd_data_train[metric_plot] >= val_min]
            pd_data_dev_plot = pd_data_dev.loc[pd_data_dev[metric_plot] >= val_min]
            pd_data_test_plot = pd_data_test.loc[pd_data_test[metric_plot] >= val_min]
            pd_data_test_extra_plot = pd_data_test_extra.loc[pd_data_test_extra[metric_plot] >= val_min]
            ax[i_plot, 0].plot(pd_data_train_plot[metric_plot], pd_data_train_plot['log_growth_mc'], '.')
            ax[i_plot, 0].set_xlabel('Predicted Growth mean min')
            ax[i_plot, 0].set_ylabel('Actual Growth')
            ax[i_plot, 0].set_title('Training set')

            ax[i_plot, 1].plot(pd_data_dev_plot[metric_plot], pd_data_dev_plot['log_growth_mc'], '.')
            ax[i_plot, 1].set_xlabel('Predicted Growth mean min')
            ax[i_plot, 1].set_ylabel('Actual Growth')
            ax[i_plot, 1].set_title('Dev set')

            pd_data_test_plot_3 = pd_data_test_plot.loc[pd_data_test_plot.num_p == 3]
            pd_data_test_plot_4 = pd_data_test_plot.loc[(pd_data_test_plot.num_p == 4)]
            ax[i_plot, 2].plot(pd_data_test_plot_4[metric_plot], pd_data_test_plot_4['log_growth_mc'], '.', label='4')
            ax[i_plot, 2].plot(pd_data_test_plot_3[metric_plot], pd_data_test_plot_3['log_growth_mc'], '.', label='3')
            ax[i_plot, 2].set_xlabel('Predicted Growth mean min')
            ax[i_plot, 2].set_ylabel('Actual Growth')
            ax[i_plot, 2].set_title('Testing set')
            ax[i_plot, 2].legend(loc=4)

            pd_data_test_extra_plot_1 = pd_data_test_extra_plot.loc[pd_data_test_extra_plot.num_p == 1]
            pd_data_test_extra_plot_2 = pd_data_test_extra_plot.loc[pd_data_test_extra_plot.num_p == 2]
            ax[i_plot, 3].plot(pd_data_test_extra_plot_2[metric_plot], pd_data_test_extra_plot_2['log_growth_mc'], '.', label='2')
            ax[i_plot, 3].plot(pd_data_test_extra_plot_1[metric_plot], pd_data_test_extra_plot_1['log_growth_mc'], '.', label='1')
            ax[i_plot, 3].set_xlabel('Predicted Growth mean min')
            ax[i_plot, 3].set_ylabel('Actual Growth')
            ax[i_plot, 3].set_title('Extra testing set')
            ax[i_plot, 3].legend(loc=4)
        fig.tight_layout()

        pd_regr_train = pd.DataFrame({'y': pd_data_train_plot['log_growth_mc'], 'num_p': pd_data_train_plot['num_p'],
                                      'y_pred': pd_data_train_plot[metric_plot], 'rdq': pd_data_train_plot['rdq_0']})
        pd_regr_dev = pd.DataFrame({'y': pd_data_dev_plot['log_growth_mc'], 'num_p': pd_data_dev_plot['num_p'],
                                    'y_pred': pd_data_dev_plot[metric_plot], 'rdq': pd_data_train_plot['rdq_0']})
        pd_regr_extra_test = pd.DataFrame({'y': pd_data_test_extra_plot['log_growth_mc'], 'num_p': pd_data_test_extra_plot['num_p'],
                                           'y_pred': pd_data_test_extra_plot[metric_plot],
                                           'rdq': pd_data_train_plot['rdq_0']})
        thresholds = [0.03, 0.04, 0.05, 0.075, 0.1, 0.15]
        n_fig = len(thresholds)
        fig, ax = plt.subplots(3, n_fig, figsize=(18, 9))
        ax = fig.axes
        for j, dataset_type in enumerate(['Dev', 'Test', 'Extra Test']):
            if dataset_type == 'Dev':
                pd_dataset = pd_data_dev_plot
            elif dataset_type == 'Test':
                pd_dataset = pd_data_test_plot
            else:
                pd_dataset = pd_data_test_extra_plot
            pd_regr = pd.DataFrame({'y': pd_dataset['log_growth_mc'], 'num_p': pd_dataset['num_p'],
                                    'y_pred': pd_dataset[metric_plot], 'rdq': pd_dataset['rdq_0']})
            for i, threshold in enumerate(thresholds):
                if i < (len(thresholds) - 1):
                    pd_select = pd_regr.loc[(pd_regr.y_pred >= thresholds[i]) & (pd_regr.y_pred < thresholds[i + 1])]
                    title = f'Between {thresholds[i]} and {thresholds[i + 1]} \n mean {round(pd_select.y.mean(), 3)}'
                else:
                    pd_select = pd_regr.loc[(pd_regr.y_pred >= thresholds[i])]
                    title = f'> {threshold} \n mean {round(pd_select.y.mean(), 3)}'
                bins = np.histogram(pd_select.y, bins=20)[1]
                for num_p in list(pd_select.num_p.unique()):
                    for direction in ['<', '>']:
                        if direction == '<':
                            pd_select_hist = pd_select.loc[(pd_select.num_p <= num_p) & (pd_select.rdq.str[:4] <= '2010')]
                        else:
                            pd_select_hist = pd_select.loc[(pd_select.num_p <= num_p) & (pd_select.rdq.str[:4] > '2010')]
                        if len(pd_select_hist) > 0:
                            ax[j * len(thresholds) + i].hist(pd_select_hist.y, bins=bins, label=f'{int(num_p)}-{direction}')
                ax[j * len(thresholds) + i].set_title(f'{dataset_type}\n{title}')
                if len(pd_select) > 0:
                    ax[j * len(thresholds) + i].legend()
        fig.tight_layout()

if 1 == 0:
    pd_data_exe = pd_data_ori.loc[pd_data_ori.num_p.isna()].copy()
    # pd_data_exe = pd_data_exe.loc[((pd_data_exe.marketcap_0 / pd_data_exe.book_value_0) <= mc_book_ratio[1]) &
    #                               ((pd_data_exe.marketcap_0 / pd_data_exe.book_value_0) >= mc_book_ratio[0])]
    # pd_data_exe = pd_data_exe.loc[pd_data_exe.marketcap_0 >= marketcap_min].copy()
    # pd_data_exe['rdq_0'] = pd.read_sql("select max(time) as time from price where symbol = 'AAPL'", con).iloc[0].time
    pd_marketcap_exe_latest = stock_price.get_marketcap_latest(list(pd_data_exe.symbol))[['symbol', 'marketcap_latest']]
    merge_cols = ['symbol']
    pd_marketcap_current_exe_1 = pd_data_exe[merge_cols].merge(pd_marketcap_exe_latest, on=merge_cols, how='left')
    pd_data_exe['marketcap_0'] = list(pd_marketcap_current_exe_1['marketcap_latest'])

    log_grow_pred_exe_mean, log_grow_pred_exe_std = get_prediction(pd_data_exe, dict_transform, regr_list_global)
    pd_data_exe['log_growth_mc_pred_mean'] = log_grow_pred_exe_mean
    pd_data_exe['log_growth_mc_pred_std'] = log_grow_pred_exe_std
    pd_data_exe['log_growth_mc_pred_min'] = log_grow_pred_exe_mean - log_grow_pred_exe_std * 2
    head_keys = ['symbol', 'datafqtr', 'log_growth_mc_pred_min', 'log_growth_mc_pred_mean', 'log_growth_mc_pred_std']
    pd_data_exe = pd_data_exe[head_keys + [i for i in pd_data_exe.columns if i not in head_keys]]

    pd_data_exe = pd_data_exe.sort_values('log_growth_mc_pred_min', ascending=False).copy()
    pd_data_exe.index = np.arange(len(pd_data_exe))

    budget = 10211
    leverage = 0.99
    metric_min = 0.03
    n_stocks_max = 7

    pd_data_exe_view = pd_data_exe.loc[pd_data_exe.log_growth_mc_pred_min >= metric_min].head(n_stocks_max)
    n_stocks = len(pd_data_exe_view)
    pd_prices = stock_price.get_price_latest(list(pd_data_exe.symbol.unique()))
    pd_allocate = pd_prices[['symbol', 'adjclose']].copy()
    _budget = budget * leverage / n_stocks
    pd_allocate['num'] = (_budget / pd_allocate['adjclose']).astype(int)
    pd_allocate = pd_data_exe_view[['symbol', 'log_growth_mc_pred_min', 'log_growth_mc_pred_mean', 'log_growth_mc_pred_std',
                                    'marketcap_0']].merge(pd_allocate, on='symbol', how='inner')
    pd_allocate = pd_allocate.sort_values('log_growth_mc_pred_min', ascending=False)
    margin_rate = budget / (pd_allocate['num'] * pd_allocate['adjclose']).sum()

    symbols_last = ['BOMN', 'SHOP', 'SQ', 'SPWH', 'CHGG', 'EVBG', 'DXCM']
    pd_data_exe_last = pd_data_exe.loc[pd_data_exe.symbol.isin(symbols_last)]

if 1 == 0:
    import numpy as np

    a = np.asarray([0.1, -0.1, 0.15, 0.16, 0.2, 0.15, 0.13])
    b1, b2 = round(10 ** (a.mean()), 5), round((10 ** a).mean(), 5)
    c1, c2 = round(np.log10(b1), 5), round(np.log10(b2), 5)
    print(b1, b2)
    print(c1, c2)

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