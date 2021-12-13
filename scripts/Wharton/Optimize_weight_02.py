__author__ = 'Yunsong Xie'
__email__ = 'xiefinance00@gmail.com'
__company__ = 'Xie Finance LLC'

import re, os, sys, datetime, sqlite3
import numpy as np
import pandas as pd
import time, glob, threading
import queue
import concurrent.futures
from matplotlib import pyplot as plt
import lib as common_func
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost, scipy
import multiprocessing as mp
import pickle

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

    def prepage_training_data(pd_data_train, ):
        """

        Args:
            pd_data_train:
            :

        """
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
            p_feature = 'marketcap_p'
        elif data_type == 'investing':
            p_feature = None
        else:
            raise ValueError('datatype has to be in [training, evaluation, investing]')
        pd_mdata = pd_data[['datatype']].copy()
        if p_feature:
            time_col = f'rdq_{p_feature.split("_")[-1]}'
            pd_mdata[time_col] = pd_data[time_col]
            if (coeff_fade > 0) & (coeff_fade <= 1):
                pd_mdata['weight'] = coeff_fade ** (np.abs(pd.to_datetime(pd_mdata[time_col]) -
                                                           pd.to_datetime(pd_mdata[time_col]).max()).dt.days / 365)
            else:
                raise ValueError('coeff_fade has to be a positive number that is smaller 1 ')
            pd_mdata['mc_growth'] = pd_data[p_feature] / pd_data['marketcap_0']
            pd_mdata['mc_growth_log'] = list(np.log10(pd_mdata['mc_growth']))
            pd_mdata['mc_growth_log_squred'] = list(pd_mdata['mc_growth_log'] ** 2)

        major_feature = dict_transform['major_feature']

        features_bvr_year = ['cur_asset', 'cur_liab', 'cash_invest', 'cash_flow', 'revenue', 'profit']
        features_growth = ['book_value', 'revenue']
        features_add = ['num', 'num_p', 'revenue_0_growth_quantile', 'book_value_0_growth_quantile']
        mc_bv_years = 3

        feature_swap = {'book_value': 'revenue', 'revenue': 'book_value'}
        if major_feature in features_bvr_year:
            features_bvr_year[features_bvr_year.index(major_feature)] = feature_swap[major_feature]

        for mc_bv_year in range(mc_bv_years):
            pd_mdata[f'mc_bv_{mc_bv_year}'] = list(pd_data[f'marketcap_{mc_bv_year}'] / pd_data[f'{major_feature}_{mc_bv_year}'])
        pd_mdata['mc_bv_q1'] = list(pd_data[f'marketcap_0'] / pd_data[f'{major_feature}_q1'])
        pd_mdata['mc_bv_q4'] = list(pd_data[f'marketcap_0'] / pd_data[f'{major_feature}_1'])
        for feature in features_add:
            pd_mdata[feature] = pd_data[feature]

        features_x = [i for i in pd_mdata.columns if 'mc_bv' in i] + [i for i in features_add if i != 'num']

        for feature in features_growth:
            for i_year in range(n_year_x - 1):
                feature_x = f'{feature}_growth_{i_year}'
                pd_mdata[feature_x] = list(pd_data[f'{feature}_{i_year}'] / pd_data[f'{feature}_{i_year + 1}'])
                features_x.append(feature_x)

            for i_quarter in [0]:
                feature_x = f'{feature}_growth_q{i_quarter}'
                pd_mdata[feature_x] = list(pd_data[f'{feature}_q{i_quarter}'] / pd_data[f'{feature}_q{i_quarter + 4}'])
                features_x.append(feature_x)

        for i_year in range(n_year_x):
            for feature in features_bvr_year:
                feature_x = f'bvr_{feature}_{i_year}'
                pd_mdata[feature_x] = list(pd_data[f'{feature}_{i_year}'] / pd_data[f'{major_feature}_{i_year}'])
                features_x.append(feature_x)
        for i_quarter in [0]:
            for feature in features_bvr_year:
                feature_x = f'bvr_{feature}_q{i_quarter}'
                pd_mdata[feature_x] = list(pd_data[f'{feature}_q{i_quarter}'] / pd_data[f'{major_feature}_q{i_quarter}'])
                features_x.append(feature_x)

        return pd_mdata, features_x

    def get_model_sklearn(pd_train, pd_pseudo, dict_transform, seed=None):

        if seed is not None:
            np.random.seed(seed)
        func_shift, func_power = dict_transform['func_shift'], dict_transform['func_power']

        _pd_mdata_train, features_x = prepare_features(pd_train, dict_transform, data_type='training')
        _pd_mdata_pseudo, features_x = prepare_features(pd_pseudo, dict_transform, data_type='evaluation')
        pd_mdata = pd.concat([_pd_mdata_train, _pd_mdata_pseudo]).copy()

        features_bvr_year = ['cur_asset', 'cur_liab', 'cash_invest', 'cash_flow', 'revenue', 'profit']
        features_growth = ['book_value', 'revenue']
        features_x_select = ['mc_bv_0', 'mc_bv_1', 'num_p', 'revenue_0_growth_quantile', 'book_value_0_growth_quantile']

        features_bvr_year = ['cur_asset', 'cur_liab', 'cash_invest', 'cash_flow', 'revenue', 'profit']
        features_bvr_year_label = [0]
        features_bvr_quarter = ['cur_asset', 'cur_liab']

        features_growth = ['book_value', 'revenue']
        features_growth_year_label = [0]
        features_x_select = ['num_p', 'mc_bv_0']
        features_growth_time_label = ['year']


        for _ in features_growth:
            # features_x_select += [i for i in features_x if (_ in i) & ('growth' in i) & ('q4' not in i)]
            # features_x_select += [i for i in features_x if (_ in i) & ('growth' in i)]
            if 'year' in features_growth_time_label:
                for year_label in [str(i) for i in features_growth_year_label]:
                    features_x_select += [i for i in features_x if (_ in i) & ('growth' in i) & ('quantile' not in i) &
                                          ('q' not in i) & (year_label in i)]
            if 'quarter' in features_growth_time_label:
                features_x_select += [i for i in features_x if (_ in i) & ('growth' in i) & ('quantile' not in i) & ('q' in i) & ('0' in i)]
        for _ in features_bvr_year:
            for year_label in [str(i) for i in features_bvr_year_label]:
                features_x_select += [i for i in features_x if (_ in i) & ('bvr' in i) & ('q' not in i) & (year_label in i)]
        for _ in features_bvr_quarter:
            features_x_select += [i for i in features_x if (_ in i) & ('bvr' in i) & ('q' in i) & ('0' in i)]

        # pd_mdata_norm = _pd_mdata_train
        pd_mdata_norm = pd_mdata
        for feature in features_x:
            if feature in dict_transform['features_exempt']:
                dict_transform['mean'][feature] = 0
                dict_transform['std'][feature] = 1
            else:
                ind_neg_train = pd_mdata_norm[feature] <= 0
                if any(ind_neg_train):
                    pd_mdata_norm.loc[ind_neg_train, feature] = pd_mdata_norm.loc[pd_mdata_norm[feature] > 0, feature].min()
                ind_neg_all = pd_mdata[feature] <= 0
                if any(ind_neg_all):
                    pd_mdata.loc[ind_neg_all, feature] = pd_mdata.loc[pd_mdata[feature] > 0, feature].min()

                col = np.log10(pd_mdata_norm[feature].values)
                mean, std = col.mean(), col.std()
                dict_transform['mean'][feature] = mean
                dict_transform['std'][feature] = std
                pd_mdata[feature] = (np.log10(pd_mdata[feature]) - mean) / std / dict_transform['std_adjust']

        pd_estimator = pd.DataFrame({'estimator': dict_transform['n_estimators_list'], 'learning_rate': dict_transform['learning_rates'],
                                     'max_depth': dict_transform['max_depth_list'], 'subsample': dict_transform['subsample_list']})
        pd_estimator['state'] = np.random.randint(9999999, size=len(pd_estimator))
        pd_estimator = pd_estimator.iloc[sorted(range(len(pd_estimator)), key=lambda x: np.random.random())]

        y_train_save, y_median_save, y_std_save = y_transform(pd_mdata['mc_growth_log'], 'encode', func_shift, func_power, dict_transform)
        dict_transform['y_median'] = y_median_save
        dict_transform['y_std'] = y_std_save
        regr_list, time_start = [], time.time()
        n_threads = dict_transform['n_threads']
        n_regr_loop = int(np.ceil(len(pd_estimator) / n_threads))

        dict_transform['features_x'] = features_x
        dict_transform['features_x_select'] = features_x_select
        keys_thread = ['aug_size_train', 'aug_sigma_train', 'aug_size_pseudo', 'aug_sigma_pseudo', 'features_exempt', 'std_adjust',
                       'features_x_select', 'tree_method', 'predictor', 'booster', 'min_samples_split', 'min_samples_leaf', 'regr_type',
                       'bool_pseudo', 'func_shift', 'func_power']
        dict_transform_thread = {i: dict_transform[i] for i in keys_thread}

        mp_queue, n_threads = mp.Queue(), dict_transform['n_threads']

        for i_regr in range(n_threads):
            pd_estimator_thread = pd_estimator.iloc[(i_regr * n_regr_loop): ((i_regr + 1) * n_regr_loop)]
            if dict_transform['n_threads'] > 1:
                mp.Process(target=sklearn_training_thread, args=(pd_mdata, pd_estimator_thread,
                                                                 dict_transform_thread, i_regr, mp_queue)).start()
            else:
                for i_thread in range(len(pd_estimator)):
                    pd_estimator_thread = pd_estimator.iloc[[i_regr]]
                    regr_list += sklearn_training_thread(pd_mdata, pd_estimator_thread, dict_transform_thread)
                    time_span = round(time.time() - time_start, 1)
                    print(f'\rCompleted regression {i_thread + 1}/{len(pd_estimator)} - Time {time_span} s', end='')

        if dict_transform['n_threads'] > 1:
            return_count, regr_list, time_start = 0, [], time.time()
            while return_count < n_threads:
                time.sleep(0.1)
                if not mp_queue.empty():
                    _result = mp_queue.get()
                    regr_list += _result
                    return_count += 1
                time_span = time.time() - time_start
                if time_span > 30:
                    raise ChildProcessError('Something went wrong')
        print()
        return dict_transform, regr_list

    def sklearn_training_thread(pd_mdata, pd_estimator_thread, dict_transform_thread, i_queue=None, mp_queue=None):

        def add_aug_data(pd_mdata, dict_transform_thread, datatype):
            pd_mdata_cal = pd_mdata.loc[pd_mdata.datatype == datatype]
            aug_size, aug_sigma = dict_transform_thread[f'aug_size_{datatype}'], dict_transform_thread[f'aug_sigma_{datatype}']
            n_extra = aug_size * len(pd_mdata_cal)
            if aug_size:
                pd_mdata_cal_aug = pd.concat([pd_mdata_cal for _ in range(int(np.ceil(aug_size)))])
                pd_mdata_cal_aug = pd_mdata_cal_aug.iloc[:n_extra].copy()
                for feature in dict_transform_thread['features_x_select']:
                    if feature not in dict_transform_thread['features_exempt']:
                        coeff = np.random.randn(len(pd_mdata_cal_aug)) * aug_sigma / dict_transform_thread['std_adjust']
                        pd_mdata_cal_aug[feature] = pd_mdata_cal_aug[feature] + coeff
                pd_mdata_cal = pd.concat([pd_mdata_cal, pd_mdata_cal_aug])
            return pd_mdata_cal

        features_x_select = dict_transform_thread['features_x_select']
        pd_mdata_cal_train = add_aug_data(pd_mdata, dict_transform_thread, datatype='train')
        if dict_transform_thread['bool_pseudo']:
            pd_mdata_cal_pseudo = add_aug_data(pd_mdata, dict_transform_thread, datatype='pseudo')
            x_pseudo = pd_mdata_cal_pseudo[features_x_select].values
            weight_pseudo = pd_mdata_cal_pseudo['weight'].values
        else:
            x_pseudo, weight_pseudo = None, None

        tree_method, predictor, booster = dict_transform_thread['tree_method'], dict_transform_thread['predictor'], dict_transform_thread['booster']
        func_shift, func_power = dict_transform_thread['func_shift'], dict_transform_thread['func_power']

        x_train, y_train_ori = pd_mdata_cal_train[features_x_select].values, pd_mdata_cal_train['mc_growth_log'].values
        y_train, y_median, y_std = y_transform(pd_mdata_cal_train['mc_growth_log'], 'encode', func_shift, func_power, dict_transform_thread)
        weight_train = pd_mdata_cal_train['weight'].values


        regr_list = []

        for i_regr in range(len(pd_estimator_thread)):
            n_estimators, learning_rate = pd_estimator_thread.iloc[i_regr][['estimator', 'learning_rate']]
            subsample, max_depth = pd_estimator_thread.iloc[i_regr][['subsample', 'max_depth']]
            min_samples_split, min_samples_leaf = dict_transform_thread['min_samples_split'], dict_transform_thread['min_samples_leaf']
            n_estimators, max_depth = int(n_estimators), int(max_depth)

            state1 = int(pd_estimator_thread.iloc[i_regr]['state'])
            state2 = state1 + np.random.randint(99999)
            if dict_transform_thread['regr_type'] == 'xgboost_GB':
                regr1 = xgboost.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                              predictor=predictor, tree_method=tree_method, random_state=state1)
                regr2 = xgboost.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                              predictor=predictor, tree_method=tree_method, random_state=state2)
            elif dict_transform_thread['regr_type'] == 'xgboost_RF':
                regr1 = xgboost.XGBRFRegressor(n_estimators=n_estimators, num_parallel_tree=n_estimators, subsample=subsample,
                                              max_depth=max_depth, learning_rate=1, booster=booster,
                                              predictor=predictor, random_state=state1, n_jobs=6)
                regr2 = xgboost.XGBRFRegressor(n_estimators=n_estimators, num_parallel_tree=n_estimators, subsample=subsample,
                                              max_depth=max_depth, learning_rate=1, booster=booster,
                                              predictor=predictor, random_state=state2, n_jobs=6)
            elif dict_transform_thread['regr_type'] == 'sklearn_RF':
                regr1 = RandomForestRegressor(n_estimators=n_estimators, max_samples=subsample, max_depth=max_depth, random_state=state1,
                                              n_jobs=-1, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                regr2 = RandomForestRegressor(n_estimators=n_estimators, max_samples=subsample, max_depth=max_depth, random_state=state2,
                                              n_jobs=-1, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            else:
                raise KeyError('regr_type can only be [GB, RF]')
            regr1.fit(x_train, y_train, sample_weight=weight_train)
            if dict_transform_thread['bool_pseudo']:
                y_pseudo = regr1.predict(x_pseudo)
                x_final, y_final = np.concatenate([x_train, x_pseudo]), np.concatenate([y_train, y_pseudo])
                weight_final = np.concatenate([weight_train, weight_pseudo])
                regr2.fit(x_final, y_final, sample_weight=weight_final)
                regr_list.append(regr2)
            else:
                regr_list.append(regr1)
            # print(f'Completed regression thread - {i_queue} - {i_regr + 1}/{len(pd_estimator_thread)}')

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

        for feature in [i for i in dict_transform['mean'] if i in pd_mdata.columns]:
            ind_neg = pd_mdata[feature] <= 0
            if any(ind_neg):
                pd_mdata.loc[ind_neg, feature] = pd_mdata.loc[pd_mdata[feature] > 0, feature].min()
            col = np.log10(pd_mdata[feature].values)
            mean, std = dict_transform['mean'][feature], dict_transform['std'][feature]
            col = (col - mean) / std / dict_transform['std_adjust']
            pd_mdata[feature] = col

        pd_mdata[features_x] = pd_mdata[features_x].fillna(0)
        X_cal = pd_mdata[features_x].values

        # y_ori = pd_mdata['mc_growth_log'].values
        # y_actual, y_median, y_std = y_transform(y_ori, 'encode', func_shift, func_power, dict_transform)
        aug_sigma_pred = dict_transform['aug_sigma_pred']
        if dict_transform['aug_size_pred'] > 0:
            x_array_aug = np.concatenate([X_cal for _ in range(dict_transform['aug_size_pred'])], axis=0)
            coeff_array = np.random.randn(x_array_aug.shape[0], x_array_aug.shape[1]) * aug_sigma_pred / dict_transform['std_adjust']
            for feature in [i for i in dict_transform['features_exempt'] if i in features_x]:
                coeff_array[:features_x.index(feature)] = 0
            x_array_aug = x_array_aug + coeff_array
            x_array_pred = np.concatenate([X_cal, x_array_aug])
        else:
            x_array_pred = X_cal
        y_array_pred_list = []
        _pd_mdata_pred = pd.DataFrame(data=x_array_pred, columns=features_x)
        x_array_final = _pd_mdata_pred[dict_transform['features_x_select']]
        for i_regr, regr in enumerate(regr_list):
            y_array_as_pred = regr.predict(x_array_final)
            y_array_pred_entry = y_array_as_pred.reshape(dict_transform['aug_size_pred'] + 1, X_cal.shape[0]).T
            y_array_pred_list.append(y_array_pred_entry)
        y_pred_concat_as_pred = np.concatenate(y_array_pred_list, axis=1)
        y_pred_concat = y_transform(y_pred_concat_as_pred, 'decode', func_shift, func_power, dict_transform)
        y_pred_mean = y_pred_concat.mean(axis=1)
        y_pred_median = np.quantile(y_pred_concat, 0.5, axis=1)
        y_pred_std = y_pred_concat.std(axis=1)
        y_pred_median = y_pred_median + cal_trading_weight(pd_data, dict_transform) * dict_transform['adj_metric']
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
            if output[5:] == '02-29':
                if year % 4 != 0:
                    output = f'{year}-02-28'


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

    def add_quantile_info(pd_base, ratio_stock_select, ratio_stock_select_span_year):
        np_data = pd_base[['rdq_0', 'revenue_0_growth', 'book_value_0_growth'] + ['marketcap_0'] * 5].values
        np_data[:, 0] = pd.to_datetime(np_data[:, 0])
        np_data[:, [1, 2]] = np_data[:, [1, 2]].astype(float).round(5)
        np_data[:, 3], np_data[:, 4], dtime = np.nan, np.nan, pd.to_timedelta(f'{int(366 * ratio_stock_select_span_year)} days')
        ind_array = np.arange(len(np_data))
        for rdq_0 in sorted(sorted(np.unique(np_data[:, 0]))):
            rdq_1 = rdq_0 - dtime
            _ind_cal = (np_data[:, 0] >= rdq_1) & (np_data[:, 0] <= rdq_0)
            _ind_assign = np_data[:, 0] == rdq_0
            np_data[_ind_assign, 3] = round(np.quantile(np_data[_ind_cal, 1], 1 - ratio_stock_select), 6)
            np_data[_ind_assign, 4] = round(np.quantile(np_data[_ind_cal, 2], 1 - ratio_stock_select), 6)
            np_data[_ind_assign, 5] = sum(_ind_cal)
            for _ind in ind_array[_ind_assign]:
                quantile_revenue = sum(np_data[_ind, 1] >= np_data[_ind_cal, 1])
                np_data[_ind, 6] = int(round(quantile_revenue / np_data[_ind, 5] * 100))
                revenue_revenue = sum(np_data[_ind, 2] >= np_data[_ind_cal, 2])
                np_data[_ind, 7] = int(round(revenue_revenue / np_data[_ind, 5] * 100))
        pd_base['revenue_0_growth_threshold'] = np_data[:, 3].astype(float)
        pd_base['book_value_0_growth_threshold'] = np_data[:, 4].astype(float)
        pd_base['revenue_0_growth_quantile'] = np_data[:, 6].astype(float)
        pd_base['book_value_0_growth_quantile'] = np_data[:, 7].astype(float)
        pd_base['num_growth'] = np_data[:, 5].astype(float)
        keys = list(pd_base.keys())
        ind_end = keys.index('book_value_0_growth')
        new_keys = keys[ind_end+1:]

        return pd_base, new_keys

    def prepare_pd_data_operate(pd_base, _pd_data, new_keys_growth):
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
        pd_data_operate['year'] = pd_data_operate['rdq_0'].str[:4].astype(int)
        head_keys = ['symbol', 'datafqtr', 'num_valid', 'year']
        pd_data_operate = pd_data_operate[head_keys + [i for i in pd_data_operate.columns if i not in head_keys]]

        pd_temp = pd_base[['rank'] + new_keys_growth].copy()
        pd_temp_list = []
        for i in range(4):
            pd_temp_1 = pd_temp.copy()
            pd_temp_1['rank_ori'] = pd_temp_1['rank']
            pd_temp_1['rank'] = pd_temp_1['rank'] + i
            pd_temp_list.append(pd_temp_1)
        pd_temp_2 = pd.concat(pd_temp_list)
        pd_temp_3 = pd_temp_2.groupby('rank')['rank_ori'].min().reset_index()
        pd_base_right = pd_temp.rename(columns={'rank': 'rank_ori'}).merge(pd_temp_3, on='rank_ori', how='right')
        pd_base_right = pd_base_right[[i for i in pd_base_right.columns if i != 'rank_ori']]

        pd_data_operate = pd_data_operate.merge(pd_base_right, on='rank', how='inner')
        return pd_data_operate

    class Transaction:
        def __init__(self, dict_transform):
            self.dict_transform = dict_transform
            self.eval_metric = dict_transform['eval_metric']
            self.n_stocks, self.n_stocks_copy = dict_transform['n_stocks'] , dict_transform['n_stocks']
            self.ratio_threshold_sell, self.ratio_threshold_buy = dict_transform['ratio_threshold_sell'], dict_transform['ratio_threshold_buy']
            self.ratio_margin, self.margin_interest = dict_transform['ratio_margin'], dict_transform['margin_interest']
            self.evaluate_span_month, self.replace_span_month = dict_transform['evaluate_span_month'], dict_transform['replace_span_month']
            self.training_num_p_min, self.sell_type = dict_transform['training_num_p_min'], dict_transform['sell_type']
            self.bool_replace, capital_gain_interest = dict_transform['bool_replace'], dict_transform['capital_gain_interest']
            self.bool_rebalance, bool_metric_recalculate = dict_transform['bool_rebalance'], dict_transform['bool_metric_recalculate']
            self.ratio_max_hold = dict_transform['ratio_max_hold']

        def set_free_n_stock(self, bool):
            if bool:
                self.n_stocks += 100
            else:
                self.n_stocks = self.n_stocks_copy

        def _sell_share_simple(self, pd_fr_record, pd_holding, symbol, rdq_s, shares, operate_type):
            pd_fr_record_new = pd.DataFrame({'datatype': [operate_type], 'symbol': [symbol], 'rdq_operate': [rdq_s]})
            pd_fr_record = pd.concat([pd_fr_record, pd_fr_record_new])

            pd_holding = pd_holding.loc[pd_holding.symbol != symbol]
            return pd_fr_record, pd_holding

        def _buy_share_simple(self, pd_fr_record, pd_holding, symbol, rdq_b, value_buy, operate_type, pd_entry):

            pd_fr_record_new = pd.DataFrame({'datatype': [operate_type], 'symbol': [symbol], 'rdq_operate': [rdq_b]})
            pd_holding_new = pd.DataFrame({'symbol': [symbol]})

            pd_fr_record = pd.concat([pd_fr_record, pd_fr_record_new])
            pd_holding = pd.concat([pd_holding, pd_holding_new])

            return pd_fr_record, pd_holding

        def _sell_share_basic(self, pd_fr_record, pd_holding, symbol, rdq_s, shares, operate_type):
            if 'pandas' in str(type(rdq_s)):
                sell_date = str(rdq_s)[:10]
            elif (type(rdq_s) is str) & (len(str(rdq_s)) == 10):
                sell_date = str(rdq_s)[:10]
            else:
                raise ValueError(f'Selling data value error, got input {rdq_s}')
            if shares < 0:
                raise ValueError("Selling share can not be negative.")

            _ind = pd_holding.symbol == symbol

            pd_quote_temp = pd_holding.loc[_ind].copy()
            pd_quote_temp['rdq_0'] = sell_date
            pd_quote = stock_price.get_marketcap_time(pd_quote_temp, time_col='rdq_0')

            # Add selling stock to record
            free_cash_current = pd_holding.iloc[0].shares

            rdq_0, num_p, cost, shares_total = pd_holding.loc[_ind].iloc[0][['rdq_0', 'num_p', 'cost', 'shares']]
            if shares > shares_total:
                raise ValueError(f"Selling shares can't be more than current holding amount {symbol}, "
                                 f"trying to sell {shares}, current holding {shares_total}")

            stock_value = shares_total * pd_quote.iloc[0].marketcap
            stock_value_sell = stock_value * shares / shares_total
            free_cash_current += stock_value_sell

            if shares < shares_total * 0.99999:
                pd_holding.loc[pd_holding.symbol == symbol, 'shares'] = shares_total * (1 - stock_value_sell / stock_value)
                pd_holding.loc[pd_holding.symbol == symbol, 'cost'] = cost * (1 - stock_value_sell / stock_value)
            else:
                pd_holding = pd_holding.loc[pd_holding.symbol != symbol].copy()

            pd_fr_record_last_buy = pd_fr_record.loc[(pd_fr_record.symbol == symbol) & (pd_fr_record.rdq_0 == rdq_0) &
                                                     (pd_fr_record.num_p == num_p) & (~pd_fr_record.datatype.str.contains('sell'))].copy()

            pd_fr_record_last_buy = pd_fr_record_last_buy.iloc[[-1]].copy()
            pd_fr_record_last_buy['rdq_operate'] = str(sell_date)[:10]
            pd_fr_record_last_buy['datatype'] = operate_type
            pd_fr_record_last_buy['cost'] = cost * (stock_value_sell / stock_value)
            pd_fr_record_last_buy['c_return'] = stock_value_sell
            pd_fr_record_last_buy['rdq_0_1st'] = str(pd_quote_temp.rdq_0_1st.iloc[0])[:10]
            pd_fr_record = pd.concat([pd_fr_record, pd_fr_record_last_buy])
            pd_holding.iloc[0] = ['free_cash', free_cash_current] + [None] * (len(pd_holding.keys()) - 2)
            return pd_fr_record, pd_holding

        def _buy_share_basic(self, pd_fr_record, pd_holding, symbol, rdq_b, value_buy, operate_type, pd_entry):
            """
            Args:
                pd_fr_record:
                pd_holding:
                symbol:
                rdq_b:
                value_buy:
                operate_type:
                pd_entry (pandas.core.series.Series/dict):
                    should contain the followed keys: symbol, rdq_0, rdq_pq4, eval_metric, num_p,
                                                      rdq_0_1st(only for rebalance)
            Returns:
            """
            bool_execute, rdq_0_1st = False, None
            if value_buy < 0:
                raise ValueError("value_buy can not be negative value.")
            if (len(pd_holding) >= (self.n_stocks + 1)) & (symbol not in list(pd_holding.symbol)):
                raise ValueError(f"Currently Holding {len(pd_holding) - 1} stocks, can't further purchase {symbol}\n"
                                 f"Holidng stocks: {', '.join(list(pd_holding.symbol.iloc[1:]))}")
            if pd_entry is not None:
                if pd_entry['symbol'] != symbol:
                    raise ValueError(f"Symbol of input pd_entry '{pd_entry['symbol']}' does not match with symbol input '{symbol}'")

            if (symbol in list(pd_holding.symbol)) & (value_buy == 0):
                # Just update the parameters
                ind_array = pd_holding.symbol == symbol
                rdq_0_1st, _shares, _cost, eval_metric_quantile = pd_holding.loc[ind_array].iloc[0][
                    ['rdq_0_1st', 'shares', 'cost', 'eval_metric_quantile']]
                pd_holding.loc[ind_array] = [symbol, _shares, rdq_0_1st, rdq_b, pd_entry['rdq_pq4'],
                                             pd_entry[self.eval_metric], pd_entry['num_p'], _cost, eval_metric_quantile]
                bool_execute = True
            else:
                # Could be rebalance or new purchase, only difference is how to calculated the previous cost
                free_cash = pd_holding.iloc[0].shares
                pd_quote_temp = pd.DataFrame({'symbol': [symbol], 'rdq_0': [str(rdq_b)[:10]]})
                pd_quote = stock_price.get_marketcap_time(pd_quote_temp, time_col='rdq_0')
                marketcap_b = pd_quote.iloc[0].marketcap

                pd_value_cal = get_holding_value(pd_holding, rdq_b, bool_keep=False)
                _value_total = pd_value_cal.value.sum()
                _ratio_margin = 0 if self.bool_rebalance else self.ratio_margin
                if (value_buy - free_cash) * 0.99999 > ((_ratio_margin + 1) * _value_total):
                    raise ValueError(f"Too much margin applied current holding \n {pd_value_cal} \n "
                                     f"Further want to purchase '{symbol}' value {round(value_buy)}")

                if value_buy >= 0:
                    pd_holding.loc[pd_holding.symbol == 'free_cash', 'shares'] = free_cash - value_buy
                    if symbol not in list(pd_holding.symbol):
                        pd_holding_new = pd.DataFrame({'symbol': [symbol], 'shares': [value_buy / marketcap_b],
                                                       'rdq_0_1st': pd_entry['rdq_0'], 'rdq_0': pd_entry['rdq_0'],
                                                       'rdq_pq4': [pd_entry['rdq_pq4']], 'pred': [pd_entry[self.eval_metric]],
                                                       'num_p': [pd_entry['num_p']], 'cost': value_buy,
                                                       'eval_metric_quantile': pd_entry.eval_metric_quantile})
                        pd_holding = pd.concat([pd_holding, pd_holding_new])

                    else:
                        # rebalance buy
                        cost_pre, shares_pre = pd_holding.loc[pd_holding.symbol == symbol].iloc[0][['cost', 'shares']]
                        shares_new = value_buy / marketcap_b
                        pd_holding.loc[pd_holding.symbol == symbol, ['cost', 'shares']] = cost_pre + value_buy, shares_pre + shares_new

                    bool_execute = True
                    rdq_0_1st = rdq_b

            if bool_execute:
                if type(pd_entry) is pd.core.series.Series:
                    _pd_temp = pd_entry.to_frame().T.copy()
                    eval_metric_quantile = pd_entry.eval_metric_quantile
                elif type(pd_entry) is dict:
                    _pd_temp = pd.DataFrame({i: [pd_entry[i]] for i in pd_entry})
                    eval_metric_quantile = pd_entry['eval_metric_quantile']
                else:
                    raise TypeError(f'Not able to recognize type of input pd_entry {type(pd_entry)}')

                _pd_temp['datatype'] = operate_type
                _pd_temp['rdq_0_1st'] = str(rdq_0_1st)[:10]
                _pd_temp['eval_metric_quantile'] = eval_metric_quantile

                pd_fr_record = pd.concat([pd_fr_record, _pd_temp])

            return pd_fr_record, pd_holding

        def sell_share(self, pd_fr_record, pd_holding, symbol, rdq_s, operate_type, bool_simple=False):

            symbols_holding = list(pd_holding.iloc[1:].symbol)
            if (symbol is not None) & (symbol not in symbols_holding):
                raise ValueError(f"{symbol} is not in pd_holding:\n{pd_holding}")
            elif bool_simple:
                pd_fr_record, pd_holding = self._sell_share_simple(pd_fr_record, pd_holding, symbol, rdq_s, 0, operate_type)

            elif (symbol in symbols_holding) & (operate_type in ['sell', 'sell_blind', 'sell_replace']):
                shares = pd_holding.loc[pd_holding.symbol == symbol].iloc[0].shares
                pd_fr_record, pd_holding = self._sell_share_basic(pd_fr_record, pd_holding, symbol,
                                                                  rdq_s, shares, operate_type)
                if self.bool_rebalance & (operate_type not in ['sell_replace']):
                    pd_fr_record, pd_holding = self.buy_share(pd_fr_record, pd_holding, pd_entry=None,
                                                              rdq_b=rdq_s, operate_type='buy_rebalance')
            elif (symbol is None) & (operate_type == 'sell_rebalance'):
                # for sell_rebalance, symbol should be None
                n_holding = len(symbols_holding)
                if n_holding > 0:
                    # Each stock sell the same percentage for cleaning up space for new stock purchase
                    pd_value_cal = get_holding_value(pd_holding, rdq_s, bool_keep=False)
                    free_cash = pd_value_cal.iloc[0].value
                    _value_total = pd_value_cal.value.sum()
                    free_cash_need = min(_value_total * self.ratio_max_hold, _value_total / (n_holding + 1))
                    value_sell_total = free_cash_need - free_cash

                    if value_sell_total > 0:
                        ratio_sell_each = value_sell_total / pd_value_cal.iloc[1:].value.sum()
                        for symbol in symbols_holding:
                            shares_holding = pd_holding.loc[pd_holding.symbol == symbol].iloc[0]['shares']
                            shares_sell = shares_holding * ratio_sell_each
                            pd_fr_record, pd_holding = self._sell_share_basic(pd_fr_record, pd_holding, symbol, rdq_s,
                                                                              shares=shares_sell, operate_type='sell_rebalance')
            else:
                raise ValueError(f"Not able to recognize this operation, sell can only be used under two cases:\n"
                                 f"1. symbol is one in pd_holding, representing clean sell, operate_type should be in ['sell', 'sell_blind']\n"
                                 f"2. symbol is None operate_type is sell_rebalance. \n"
                                 f"Input symbol is '{symbol}', operate_type is '{operate_type}'")
            return pd_fr_record, pd_holding

        def buy_share(self, pd_fr_record, pd_holding, pd_entry, rdq_b, operate_type, value_buy_force=None, bool_simple=False):
            """
            Args:
                pd_fr_record:
                pd_holding:
                pd_entry (pandas.core.series.Series/dict):
                    should contain the followed keys symbol, rdq_0, rdq_pq4, eval_metric, num_p
                rdq_b:
                operate_type:
                value_buy_force (float/int): The exact buy value for this operation
                bool_simple (boolean): Whether to do a simple holding change
            Returns:
            """

            if len(pd_holding) >= (self.n_stocks + 1):
                # too many stock is being held, no purchase to be executed
                return pd_fr_record, pd_holding

            if bool_simple:
                symbol = pd_entry['symbol']
                pd_fr_record, pd_holding = self._buy_share_simple(pd_fr_record, pd_holding, symbol, rdq_b, 0, operate_type, pd_entry)

            elif (pd_entry is not None) & (operate_type in ['buy', 'buy_replace']):
                symbol = pd_entry['symbol']
                if symbol in list(pd_holding.symbol):
                    pd_fr_record, pd_holding = self._buy_share_basic(pd_fr_record, pd_holding, symbol, rdq_b, 0, 'refresh', pd_entry)
                else:
                    if self.bool_rebalance:
                        # How much cap to be cleaned is determined in sell_share function
                        if operate_type != 'buy_replace':
                            pd_fr_record, pd_holding = self.sell_share(pd_fr_record, pd_holding, symbol=None,
                                                                       rdq_s=rdq_b, operate_type='sell_rebalance')
                            pd_value_cal = get_holding_value(pd_holding, rdq_b, bool_keep=False)
                            _value_total = pd_value_cal.value.sum()
                            free_cash = pd_value_cal.iloc[0].shares
                            value_buy = min(free_cash, _value_total * self.ratio_max_hold)
                        else:
                            value_buy = value_buy_force

                        pd_fr_record, pd_holding = self._buy_share_basic(pd_fr_record, pd_holding, symbol, rdq_b,
                                                                         value_buy, operate_type, pd_entry)
                    else:
                        # ratio_margin is only applied when rebalance is deactivated
                        # ratio_margin is only used when mean value can't be achieved using the available free cash
                        pd_value_cal = get_holding_value(pd_holding, rdq_b, bool_keep=False)
                        _value_total = pd_value_cal.value.sum()
                        free_cash = pd_value_cal.iloc[0].shares
                        _value_mean = _value_total / self.n_stocks
                        if free_cash > _value_mean:
                            if len(pd_holding) == self.n_stocks:
                                value_buy = free_cash
                            else:
                                value_buy = _value_mean
                        else:
                            _margin = _value_total * self.ratio_margin
                            if free_cash > -_margin:
                                value_buy = min(free_cash + _margin, _value_mean)
                            else:
                                value_buy = 0
                        pd_fr_record, pd_holding = self._buy_share_basic(pd_fr_record, pd_holding, symbol, rdq_b,
                                                                         value_buy, operate_type, pd_entry)

            elif (pd_entry is None) & (operate_type == 'buy_rebalance'):

                if len(pd_holding) != 1:
                    free_cash = pd_holding.iloc[0].shares
                    pd_value_cal = get_holding_value(pd_holding, rdq_b, bool_keep=False)
                    _value_total = pd_value_cal.value.sum()

                    pd_value_cal['value_goal'] = _value_total * self.ratio_max_hold
                    pd_value_cal['value_add'] = pd_value_cal['value_goal'] - pd_value_cal['value']
                    pd_value_cal_add = pd_value_cal.loc[(pd_value_cal['value_add'] > 0) & (pd_value_cal['symbol'] != 'free_cash')].copy()
                    value_add_total = pd_value_cal_add['value_add'].sum()
                    value_add_final_total = min(free_cash, value_add_total)
                    pd_value_cal_add['value_add_final'] = pd_value_cal_add['value_add'] / value_add_total * value_add_final_total
                    for ind in range(len(pd_value_cal_add)):
                        pd_value_cal_add_entry = pd_value_cal_add.iloc[ind]
                        symbol, value_add = pd_value_cal_add_entry[['symbol', 'value_add_final']]
                        pd_entry = {i: pd_value_cal_add_entry[i] for i in ['symbol', 'rdq_0', 'rdq_pq4', 'pred',
                                                                           'num_p', 'eval_metric_quantile']}
                        pd_entry[self.eval_metric] = pd_value_cal_add_entry['pred']
                        pd_fr_record, pd_holding = self._buy_share_basic(pd_fr_record, pd_holding, symbol, rdq_b,
                                                                         value_add, 'buy_rebalance', pd_entry)

            else:
                raise KeyError(f"Not able to recognize this operation, sell can only be used under two cases:\n"
                               f"1. pd_entry is None, operate_type is 'buy'\n"
                               f"2. pd_entry is not None, operate_type is 'buy_rebalance'\n"
                               f"Input pd_entry is '{pd_entry}', operate_type is '{operate_type}'")

            return pd_fr_record, pd_holding

        def swap_share(self, pd_fr_record, pd_holding, pd_entry, symbol_hold, rdq_buy, bool_simple=False):
            """
            Replace share
            Args:
                pd_fr_record (pandas.dataframe): Transaction recording data
                pd_holding (pandas.dataframe): Currently held stock data
                pd_entry (dict/pandas.dataframe): Info needed for the purchasing transaction, should at least contains:
                        ['symbol', 'rdq_0', 'rdq_pq4', 'pred', 'num_p', 'eval_metric_quantile']
                symbol_hold (str): symbol of the currently held
                rdq_buy (str/datetime): date of the purchasing
                bool_simple (boolean): Whether to do a simple holding change
            Returns:
                pd_fr_record (pandas.dataframe): Renewed transaction recording data
                pd_holding (pandas.dataframe): Renewed currently held stock data
            """
            pd_fr_record, pd_holding = self.sell_share(pd_fr_record, pd_holding, symbol_hold, rdq_buy, 'sell_replace', bool_simple)
            if not bool_simple:
                value_buy_force = pd_fr_record.iloc[-1]['c_return']
                pd_fr_record, pd_holding = self.buy_share(pd_fr_record, pd_holding, pd_entry, rdq_buy, 'buy_replace',
                                                          value_buy_force, bool_simple)
            else:
                pd_fr_record, pd_holding = self.buy_share(pd_fr_record, pd_holding, pd_entry, rdq_buy, 'buy_replace', bool_simple=bool_simple)
            return pd_fr_record, pd_holding

    def invest_period_operation(pd_fr_record, pd_holding, pd_data_operate, dict_decision_time, transaction, seed=None):

        # seed = 0
        # pd_holding = pd.DataFrame({'symbol': ['free_cash'], 'shares': [10000], 'rdq_0_1st': [None], 'rdq_0': [None], 'rdq_pq4': [None],
        #                           'pred': [None], 'num_p': [None], 'cost': [None], 'eval_metric_quantile': [None]})
        dict_transform = transaction.dict_transform
        if seed is not None:
            np.random.seed(seed)
        if 'define parameters' == 'define parameters':
            decision_time_final = dict_decision_time['start']
            decision_time_final_end = dict_decision_time['end']
            decision_time_sell_can = str(pd.to_datetime(decision_time_final) - pd.to_timedelta(f'390 days'))[:10]
            decision_time_buy_can = str(pd.to_datetime(decision_time_final) - pd.to_timedelta(f'1800 days'))[:10]
            eval_metric, rate_depreciation = dict_transform['eval_metric'], dict_transform['rate_depreciation']
            rate_step_switch, n_stocks = dict_transform['rate_step_switch'], dict_transform['n_stocks']
            ratio_threshold_sell, ratio_threshold_buy = dict_transform['ratio_threshold_sell'], dict_transform['ratio_threshold_buy']
            ratio_margin, margin_interest = dict_transform['ratio_margin'], dict_transform['margin_interest']
            evaluate_span_month, replace_span_month = dict_transform['evaluate_span_month'], dict_transform['replace_span_month']
            training_num_p_min, sell_type = dict_transform['training_num_p_min'], dict_transform['sell_type']
            buy_num_p_min, sell_num_p_min = dict_transform['buy_num_p_min'], dict_transform['sell_num_p_min']
            bool_replace, capital_gain_interest = dict_transform['bool_replace'], dict_transform['capital_gain_interest']
            bool_rebalance, bool_metric_recalculate = dict_transform['bool_rebalance'], dict_transform['bool_metric_recalculate']
            hold_span_month = dict_transform['hold_span_month']

        if 'prepare_data' == 'prepare_data':

            pd_data_train_pre = pd_data_operate.loc[(pd_data_operate['rdq_0'] <= decision_time_final)]

            # prepare the data for the training data
            pd_data_train_list = []
            for tq in (np.arange(4) + 1)[::-1]:
                if tq == 4:
                    _pd_temp = pd_data_train_pre.loc[(pd_data_train_pre[f'rdq_pq{tq}'] <= decision_time_final)].copy()
                else:
                    _pd_temp = pd_data_train_pre.loc[((pd_data_train_pre[f'rdq_pq{tq + 1}'] >= decision_time_final) |
                                                      pd_data_train_pre[f'rdq_pq{tq + 1}'].isna()) &
                                                     (pd_data_train_pre[f'rdq_pq{tq}'] < decision_time_final)].copy()
                _pd_temp['num'] = tq / 4
                _pd_temp['marketcap_p'], _pd_temp['rdq_p'] = _pd_temp[f'marketcap_pq{tq}'], _pd_temp[f'rdq_pq{tq}']
                pd_data_train_list.append(_pd_temp)
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
            pd_data_buy_can = pd_data_buy_can.loc[pd_data_buy_can.num_valid >= buy_num_p_min]

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
            _pd_data_sell_can = _pd_data_sell_can.loc[_pd_data_sell_can.num >= sell_num_p_min]

            if len(_pd_data_sell_can) > 0:
                # Get the latest data so that prediction can be more accurate
                pd_filter = _pd_data_sell_can.groupby('symbol').rdq_0.max().reset_index()
                pd_data_sell_can_temp = _pd_data_sell_can.merge(pd_filter, on=['symbol', 'rdq_0'], how='inner')

                # Make sure that the prediction period does NOT extend beyond the num_valid (longest extention of meeting growth standard)
                pd_data_sell_can = pd_data_sell_can_temp.loc[pd_data_sell_can_temp.num <= pd_data_sell_can_temp.num_valid]
            else:
                pd_data_sell_can = pd.DataFrame()

            pd_data_eval = pd.concat([pd_data_sell_can, pd_data_buy_can]).copy()
            head_keys = ['datatype', 'symbol', 'datafqtr', 'num_p', 'num_valid', 'num', 'marketcap_b', 'rdq_b']
            pd_data_eval['num_p'] = pd_data_eval['num']
            keys_pre = head_keys + [i for i in pd_data_eval.columns if i not in head_keys]
            pd_data_eval = pd_data_eval[[i for i in keys_pre if i in pd_data_eval.columns]]

            pd_train_pseudo = prepage_training_data(pd_data_train)
            pd_train = pd_train_pseudo.loc[pd_train_pseudo.num_p >= training_num_p_min]
            pd_pseudo = pd_train_pseudo.loc[(pd_train_pseudo.num_p < training_num_p_min) & (pd_train_pseudo.num_valid >= training_num_p_min) &
                                            (pd_train_pseudo.rdq_0 >= pd_train.rdq_0.max())].copy()
            pd_pseudo['datatype'], pd_pseudo['num_p'] = 'pseudo', pd_pseudo['num_p'] + 0.5

        bool_operature = len(pd_data_eval) > 0
        pd_data_eval_operation = []
        if 'prediction' == 'prediction':
            if not bool_operature:
                regr_list = []
            else:
                dict_transform, regr_list = get_model_sklearn(pd_train, pd_pseudo, dict_transform, seed=seed)

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

                head_keys = ['datatype', 'symbol', 'datafqtr', 'num_p', eval_metric]
                keys_pre = head_keys + [i for i in pd_train_eval.columns if i not in head_keys]
                pd_train_eval = pd_train_eval[[i for i in keys_pre if i in pd_train_eval.columns]]

                pd_data_eval_sell = pd_data_eval.loc[pd_data_eval.datatype == 'sell']
                pd_data_eval_list = []
                if len(pd_data_eval_sell) > 0:
                    pd_data_eval_sell_filter = pd_data_eval_sell.groupby('symbol').rdq_s.min().reset_index()
                    pd_data_eval_sell = pd_data_eval_sell.merge(pd_data_eval_sell_filter, on=['symbol', 'rdq_s'], how='inner')
                    pd_data_eval_sell = pd_data_eval_sell.loc[pd_data_eval_sell.symbol.isin(pd_holding.symbol)]
                    pd_data_eval_list.append(pd_data_eval_sell)

                pd_data_eval_buy = pd_data_eval.loc[(pd_data_eval.datatype == 'buy') & (pd_data_eval.num_valid == 1)]
                pd_data_eval_list.append(pd_data_eval_buy)

                # Add blind sell entries based on holding timeline
                pd_sell_blind = pd_holding.iloc[1:][['symbol', 'rdq_0']].copy()
                pd_sell_blind['rdq_s'] = (pd_sell_blind['rdq_0'] + pd.to_timedelta(f'{int(hold_span_month * 30.6)} day')).astype(str).str[:10]
                pd_sell_blind = pd_sell_blind.loc[pd_sell_blind.rdq_s <= decision_time_final_end]
                pd_sell_blind['datatype'] = 'sell_blind'
                pd_data_eval_list.append(pd_sell_blind)

                pd_data_eval_operation = pd.concat(pd_data_eval_list)

                pd_data_eval_operation['rdq_operate'] = pd_data_eval_operation['rdq_s']
                _ind = pd_data_eval_operation.datatype == 'buy'
                pd_data_eval_operation.loc[_ind, 'rdq_operate'] = pd_data_eval_operation.loc[_ind, 'rdq_b']
                head_keys = ['datatype', 'symbol', 'datafqtr', 'rdq_operate', 'rdq_0']
                pd_data_eval_operation = pd_data_eval_operation[head_keys + [i for i in pd_data_eval_operation.columns if i not in head_keys]]
                pd_data_eval_operation = pd_data_eval_operation.sort_values(by='rdq_operate')
                pd_data_eval_operation['rdq_0'] = pd.to_datetime(pd_data_eval_operation['rdq_0'])
                dict_pd_train_eval = {_num_p: pd_train_eval.loc[pd_train_eval.num_p == _num_p] for _num_p in pd_train_eval.num_p.unique()}

        if 'update_holding_metric' == 'update_holding_metric':
            if bool_metric_recalculate & (len(pd_holding) > 1):
                pd_holding_filter = pd_holding.iloc[1:][['symbol', 'rdq_0', 'num_p']].copy()
                pd_holding_filter['rdq_0'] = pd_holding_filter['rdq_0'].astype(str).str[:10]
                pd_buy_last = pd_data_operate.merge(pd_holding_filter, on=['symbol', 'rdq_0'], how='inner')
                pd_buy_last['datatype'], pd_buy_last['num'] = 'buy', pd_buy_last['num_valid']
                pd_buy_last['marketcap_b'] = pd_buy_last['marketcap_0']
                pd_buy_last['rdq_b'] = pd_buy_last['rdq_0']
                pd_buy_last['num_p'] = pd_buy_last['num_p'].astype(float)

                log_grow_pred_mean_1, log_grow_pred_median_1, log_grow_pred_std_1 = get_prediction(pd_buy_last, dict_transform, regr_list)
                if eval_metric == 'log_growth_mc_pred_min':
                    pd_holding.loc[pd_holding.symbol!='free_cash', 'pred'] = log_grow_pred_mean_1 - log_grow_pred_std_1 * 1.5
                elif eval_metric == 'log_growth_mc_pred_mean':
                    pd_holding.loc[pd_holding.symbol!='free_cash', 'pred'] = log_grow_pred_mean_1
                elif eval_metric == 'log_growth_mc_pred_median':
                    pd_holding.loc[pd_holding.symbol!='free_cash', 'pred'] = log_grow_pred_median_1

        rate_depreciation_log = np.log10(1 + rate_depreciation)
        head_keys = ['datatype', 'symbol', 'datafqtr', 'rdq_operate', 'rdq_0', 'num_p', 'num_valid', 'num', eval_metric]
        if len(pd_data_eval_operation):
            pd_data_eval_operation = pd_data_eval_operation[head_keys + [i for i in list(pd_data_eval_operation.columns) if i not in head_keys]]
            pd_data_eval_operation_show = pd_data_eval_operation.sort_values(by=['rdq_operate', 'symbol'])
            pd_data_eval_operation_show = pd_data_eval_operation_show.loc[pd_data_eval_operation_show['datatype'] == 'buy']

        for i in range(len(pd_data_eval_operation)):
            free_cash = pd_holding.iloc[0]['shares']
            pd_entry = pd_data_eval_operation.iloc[i]
            operate_type, symbol = pd_entry['datatype'], pd_entry['symbol']
            n_holding = len(pd_holding) - 1

            if 'sell' in operate_type:
                if symbol in list(pd_holding.symbol):
                    _ind = pd_holding.symbol == symbol
                    _bool_sell = False
                    if str(pd_holding.loc[_ind].iloc[0].rdq_0)[:10] < decision_time_final:
                        # Need to make sure that the time this stock bought was before this decision period
                        if operate_type == 'sell_blind':
                            _bool_sell = True
                        elif operate_type == 'sell':
                            if sell_type.lower() == 'none':
                                eval_metric_threshold_sell = - 10 ** 2
                            elif sell_type.lower() == 'ratio':
                                if ratio_threshold_sell < 0:
                                    raise ValueError(f'When sell_type is {sell_type}, ratio_threshold_sell has to be greater than 0, '
                                                     f'input is {ratio_threshold_sell}')
                                eval_metric_threshold_sell = dict_pd_train_eval[pd_entry.num_p][eval_metric].quantile(ratio_threshold_sell)
                            elif sell_type.lower() == 'rate':
                                if ratio_threshold_sell >= 0:
                                    raise ValueError(f'When sell_type is {sell_type}, ratio_threshold_sell has to be less than 0, '
                                                     f'input is {ratio_threshold_sell}')
                                eval_metric_threshold_sell = ratio_threshold_sell
                            else:
                                raise KeyError("Can't recognize sell_type value")
                            if pd_entry[eval_metric] < eval_metric_threshold_sell:
                                _bool_sell = True
                    if _bool_sell:
                        rdq_s = pd_entry['rdq_s']
                        pd_fr_record, pd_holding = transaction.sell_share(pd_fr_record, pd_holding, symbol, rdq_s, operate_type)
            elif operate_type == 'buy':
                eval_metric_value = pd_entry[eval_metric]
                rdq_buy = pd_entry['rdq_0']
                pd_temp_eval_metric = dict_pd_train_eval[pd_entry.num_p][eval_metric]
                eval_metric_threshold = pd_temp_eval_metric.quantile(ratio_threshold_buy)
                eval_metric_quantile = round(sum(eval_metric_value >= pd_temp_eval_metric) / len(pd_temp_eval_metric) * 100, 1)
                pd_entry.eval_metric_quantile = eval_metric_quantile
                _bool_buy = False

                if eval_metric_value >= eval_metric_threshold:
                    if symbol in list(pd_holding.symbol):
                        # Update the latest prediction result
                        _bool_buy = True
                    elif n_holding < n_stocks:
                        # There is free cash, buy anything that's predicted to grow more than depreciation rate
                        if (free_cash > 0) | bool_rebalance:
                            _bool_buy = True
                            # print(f'Buy {symbol}')
                    elif bool_replace:
                        # No free cash, and the symbol is not on the holding list needs to swap stocks
                        # But only swap the stock that was bought on the current period
                        replace_days = int(replace_span_month * 30)
                        rdq_replace_threshold = str(rdq_buy - pd.to_timedelta(f'{replace_days} days'))[:10]
                        _pd_pred_min = pd_holding.loc[((pd_holding.rdq_0_1st.astype(str) >= rdq_replace_threshold) |
                                                       (pd_holding.rdq_0_1st.astype(str) <= decision_time_sell_can)) &
                                                      (pd_holding.symbol != 'free_cash')]
                        if len(_pd_pred_min) > 0:
                            _argmin = np.argmin(_pd_pred_min.pred)
                            symbol_hold, _pred_min_hold = _pd_pred_min.iloc[_argmin][['symbol', 'pred']]
                            _rdq_0_hold, num_p_hold = _pd_pred_min.iloc[_argmin][['rdq_0_1st', 'num_p']]
                            _depreciation = ((rdq_buy - _rdq_0_hold).days / 365 + pd_entry.num_p - num_p_hold) * rate_depreciation_log
                            _growth_entry = 10 ** pd_entry[eval_metric]
                            _growth_hold = 10 ** _pred_min_hold * (1 + rate_step_switch) * (1 - _depreciation)
                            if _growth_entry > _growth_hold:
                                operate_type = 'replace'
                                _ind = pd_holding.symbol == symbol_hold
                                symbol_new = pd_entry['symbol']
                                # pd_fr_record, pd_holding = sell_share(pd_fr_record, pd_holding, symbol_hold, rdq_buy, 'replace')
                                pd_fr_record, pd_holding = transaction.swap_share(pd_fr_record, pd_holding, pd_entry,
                                                                                  symbol_hold, rdq_buy)
                if _bool_buy:
                    pd_fr_record, pd_holding = transaction.buy_share(pd_fr_record, pd_holding, pd_entry,
                                                                     rdq_b=pd_entry['rdq_0'], operate_type='buy')
                    _ind = (pd_data_eval_operation.datatype == 'sell_blind') & (pd_data_eval_operation.symbol == symbol)

                    if any(list(_ind)):
                        pd_data_eval_operation.loc[_ind, 'datatype'] = 'sell_blind_cancelled'

            if ((len(pd_holding) - 1) != n_holding) | (operate_type == 'replace'):
                print(i, operate_type, n_holding, pd_entry.symbol, len(pd_holding) - 1)

        if pd_holding.iloc[0].shares < 0:
            _ind = pd_holding.symbol == 'free_cash'
            _margin = pd_holding.iloc[0].shares
            pd_holding.loc[_ind, 'shares'] = _margin * (1 + margin_interest / 12 * evaluate_span_month)

        dict_transform_model = {i: dict_transform[i] for i in ['mean', 'std', 'y_median', 'y_std', 'features_x', 'features_x_select']}
        dict_transform_model['regr_list'] = regr_list
        keys_exclude = ['mean', 'std', 'y_median', 'y_std', 'regr_list']
        dict_transform_hyper = {i: dict_transform[i] for i in dict_transform.keys() if i not in keys_exclude}
        dict_transform_save = {'dict_transform_model': dict_transform_model, 'dict_transform_hyper': dict_transform_hyper}

        return pd_fr_record, pd_holding, dict_transform_save

    def cal_trading_weight(pd_data, dict_transform):
        major_feature, min_growth, max_growth = dict_transform['major_feature'], dict_transform['min_growth'], dict_transform['max_growth']
        growth_slope, max_pb_1, max_pb_2 = dict_transform['growth_slope'], dict_transform['max_pb_1'], dict_transform['max_pb_2']
        pb_slope = dict_transform['pb_slope']
        growth = pd_data[f'{major_feature}_0_growth']
        pb = pd_data[f'marketcap_0'] / pd_data[f'{major_feature}_0']
        weight_cal = (
                (1 + (np.minimum(np.maximum(growth, min_growth), max_growth) - min_growth) / (max_growth - min_growth) * growth_slope) *
                (1 + (np.maximum(np.minimum(pb, max_pb_2), max_pb_1) - max_pb_1) / (max_pb_2 - max_pb_1) * pb_slope)
                      ) - 1
        return weight_cal

    def get_holding_mrg(pd_buy_sell_thread):
        """
        First function to be executed for Multi-Run-Gathering decision making progress. This function takes the buy/sell
        decisions for each trial, reconstruct the holding status of each rdq_operate date.

        Args:
            pd_buy_sell_thread: The buy/sell decisions of this MRG holding re-construction attempt
            i_thread: The sequence of this thread
            dict_queue (dict/None): If it IS multi-thread, include two queues, 'progress', 'data', 'result', otherwise, default is None

        Returns:
            (pandas.dataframe/None): if this is NOT multi-thread, return data frame that include the holding status of each rdq_operate date
                                    if it IS  multi-thread, return None but put the data in queue, dict_queue['data']
        """
        pd_holding = pd.DataFrame({'symbol': ['free_cash'], 'shares': [10000], 'rdq_0_1st': [None],
                                   'rdq_0': [None], 'rdq_pq4': [None], 'pred': [None], 'num_p': [None], 'cost': [None],
                                   'eval_metric_quantile': [None]})
        trial_list = sorted(pd_buy_sell_thread.trial.unique())
        dict_pd_holding = {_: pd_holding.copy() for _ in trial_list}
        dict_pd_fr_record = {_: pd.DataFrame({'decision_time': []}) for _ in trial_list}
        time_start, i_rdq_operate = time.time(), 0
        rdq_operate_list = sorted(pd_buy_sell_thread.rdq_operate.unique())
        dict_pd_buy_sell = {_: pd_buy_sell_thread.loc[pd_buy_sell_thread.rdq_operate == _] for _ in rdq_operate_list}

        pd_holding_mrg_raw_list = []
        progress, n_rdq_operate = 0, len(rdq_operate_list)

        for i_rdq_operate, rdq_operate in enumerate(rdq_operate_list):
            pd_filter_1 = dict_pd_buy_sell[rdq_operate]
            _trial_list = sorted(pd_filter_1.trial.unique())
            for trial in _trial_list:
                pd_trial = pd_filter_1.loc[pd_filter_1.trial == trial]
                pd_trial_replace = pd_trial.loc[pd_trial.datatype.str.contains('replace')].sort_values(by='datatype')
                pd_trial_buy = pd_trial.loc[pd_trial.datatype == 'buy']
                pd_trial_sell = pd_trial.loc[pd_trial.datatype.isin(['sell', 'sell_blind'])]
                for i_sell in range(len(pd_trial_sell)):
                    pd_trans = pd_trial_sell.iloc[i_sell]
                    symbol, operate_type = pd_trans[['symbol', 'datatype']]
                    dict_pd_fr_record[trial], dict_pd_holding[trial] = transaction.sell_share(dict_pd_fr_record[trial], dict_pd_holding[trial],
                                                                                              symbol, rdq_operate, operate_type,
                                                                                              bool_simple=True)
                for i_buy in range(len(pd_trial_buy)):
                    pd_trans = pd_trial_buy.iloc[i_buy]
                    operate_type = pd_trans['datatype']
                    dict_pd_fr_record[trial], dict_pd_holding[trial] = transaction.buy_share(dict_pd_fr_record[trial], dict_pd_holding[trial],
                                                                                             pd_trans, rdq_operate, operate_type,
                                                                                             bool_simple=True)
                for i_replace in range(int(len(pd_trial_replace) / 2)):
                    pd_trans = pd_trial_replace.iloc[i_replace]
                    pd_trans_sell = pd_trial_replace.iloc[int(len(pd_trial_replace) / 2) + i_replace]
                    symbol_hold = pd_trans_sell['symbol']
                    dict_pd_fr_record[trial], dict_pd_holding[trial] = transaction.swap_share(dict_pd_fr_record[trial], dict_pd_holding[trial],
                                                                                              pd_trans, symbol_hold, rdq_operate,
                                                                                              bool_simple=True)
                pd_holding_save = dict_pd_holding[trial].copy()
                pd_holding_save['trial'] = trial
                pd_holding_save['rdq_operate'] = rdq_operate
                pd_holding_mrg_raw_list.append(pd_holding_save)
            time_span = round(time.time() - time_start, 1)
            progress = int(round((i_rdq_operate + 1) / n_rdq_operate * 100))
            print(f'\rTime {time_span} s - re-constructed the holding status - progress {progress}%', end='')

        pd_holding_mrg_raw_copy = pd.concat(pd_holding_mrg_raw_list)
        print(' - Completed')
        return pd_holding_mrg_raw_copy

    def find_rdq_operate(trial, rdq_operate, dict_hold):
        rdq_operate_close = None
        if rdq_operate in dict_hold[trial]:
            rdq_operate_close = rdq_operate
        else:
            keys = list(dict_hold[trial].keys())
            rdq_operate_candidates = [_ for _ in keys if _ <= rdq_operate]
            if rdq_operate_candidates:
                rdq_operate_close = max(rdq_operate_candidates)
        if rdq_operate_close:
            return dict_hold[trial][rdq_operate_close]
        else:
            return None


if __name__ == '__main__0':

    data_version = 7

    str_data_version = str(data_version).rjust(2, '0')
    dict_filename = f'result/dict_save_data_{str_data_version}.pkl'
    dict_filename_modified = f'result/dict_save_data_{str_data_version}_modified.pkl'
    if os.path.isfile(dict_filename_modified):
        with open(dict_filename_modified, 'rb') as handle:
            dict_data = pickle.load(handle)
    else:
        with open(dict_filename, 'rb') as handle:
            dict_data = pickle.load(handle)

    dict_transform = dict_data['dict_transform_hyper']
    pd_value = dict_data['pd_value']
    transaction = Transaction(dict_transform)

    pd_holding_record_final = dict_data['pd_holding_record_final']
    pd_fr_record_final = dict_data['pd_fr_record_final']

    if os.path.isfile(dict_filename_modified):
        pd_holding_mrg_pre = dict_data['pd_holding_mrg_pre']
        pd_holding_mrg_raw_copy = dict_data['pd_holding_mrg_raw_copy']
    else:
        pd_buy_sell = pd_fr_record_final.loc[pd_fr_record_final.datatype.isin(['buy', 'buy_replace', 'sell', 'sell_blind', 'sell_replace'])]
        # pd_buy_sell = pd_buy_sell.loc[~pd_buy_sell.datafqtr.isna()]
        pd_buy_sell = pd_buy_sell.sort_values(by=['rdq_operate', 'datatype', 'trial'])

        pd_holding_mrg_raw_copy = get_holding_mrg(pd_buy_sell)

        pd_holding_mrg_raw = pd_holding_mrg_raw_copy.copy()
        dict_hold = {_: {} for _ in sorted(pd_holding_mrg_raw.trial.unique())}
        trials = sorted(dict_hold.keys())
        time_start = time.time()
        for trial in trials:
            pd_data1 = pd_holding_mrg_raw.loc[pd_holding_mrg_raw.trial == trial]
            rdq_operate_list = sorted(pd_data1.rdq_operate.unique())
            for rdq_operate in rdq_operate_list:
                dict_hold[trial][rdq_operate] = pd_data1.loc[pd_data1.rdq_operate == rdq_operate]
            time_span = round(time.time() - time_start)
            print(f'\rTime {time_span} s - {trial+1}/{len(trials)} - organize trial data from pandas dataframe to dict', end='')
        print(' - completed')
        pd_holding_mrg_list = []
        time_start = time.time()
        _rdq_operate_list = sorted(pd_holding_mrg_raw.rdq_operate.unique())
        for i_rdq, rdq_operate in enumerate(_rdq_operate_list):
            for trial in sorted(pd_holding_mrg_raw.trial.unique()):
                pd_holding_found = find_rdq_operate(trial, rdq_operate, dict_hold)
                if pd_holding_found is not None:
                    pd_holding_found = pd_holding_found.copy()
                    pd_holding_found['rdq_operate_mrg'] = rdq_operate
                    pd_holding_mrg_list.append(pd_holding_found)
            time_span = round(time.time() - time_start)
            print(f'\rTime {time_span} s - {i_rdq + 1}/{len(_rdq_operate_list)} - re-map holdings for all rdq dates', end='')
        print(' - completed')
        pd_holding_mrg_pre = pd.concat(pd_holding_mrg_list)

        dict_data['pd_holding_mrg_raw_copy'] = pd_holding_mrg_raw_copy
        dict_data['pd_holding_mrg_pre'] = pd_holding_mrg_pre

        with open(dict_filename_modified, 'wb') as handle:
            pickle.dump(dict_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if 'a' == 'b':

    def modify_mrg_holding(dic_mrg_operate, rdq, symbols, action):
        for symbol in symbols:
            dic_mrg_operate['rdq'].append(rdq)
            dic_mrg_operate['symbol'].append(symbol)
            dic_mrg_operate['action'].append(action)
        return dic_mrg_operate

    min_ratio_mrg = 0.25
    pd_holding_mrg_pre = pd_holding_mrg_pre.loc[pd_holding_mrg_pre.symbol != 'free_cash']
    pd_holding_mrg = pd_holding_mrg_pre.groupby(['rdq_operate_mrg', 'symbol']).size().rename('num').reset_index()
    pd_holding_mrg['rank'] = pd_holding_mrg.groupby(['rdq_operate_mrg'])['num'].rank(method='min', ascending=False)
    pd_holding_mrg = pd_holding_mrg.loc[pd_holding_mrg['rank'] <= transaction.n_stocks_copy]
    pd_holding_mrg = pd_holding_mrg.rename(columns={'rdq_operate_mrg': 'rdq_operate'})
    pd_holding_mrg = pd_holding_mrg.loc[pd_holding_mrg.num >= np.floor(len(trials) * min_ratio_mrg)]

    transaction.set_free_n_stock(True)
    rdq_operate_list = sorted(pd_holding_mrg.rdq_operate.unique())
    dict_holding_mrg = {_: set(pd_holding_mrg.loc[pd_holding_mrg.rdq_operate == _]['symbol']) for _ in rdq_operate_list}
    dic_mrg_operate = {'rdq': [], 'symbol': [], 'action': []}
    for i_operate, rdq_operate in enumerate(rdq_operate_list):
        if i_operate == 0:
            symbols = dict_holding_mrg[rdq_operate]
            dic_mrg_operate = modify_mrg_holding(dic_mrg_operate, rdq_operate, symbols, 'buy')
        else:
            symbols_before = dict_holding_mrg[rdq_operate_list[i_operate - 1]]
            symbols_after = dict_holding_mrg[rdq_operate_list[i_operate]]
            symbols_sell = symbols_before - symbols_after
            symbols_buy = symbols_after - symbols_before
            dic_mrg_operate = modify_mrg_holding(dic_mrg_operate, rdq_operate, symbols_sell, 'sell')
            dic_mrg_operate = modify_mrg_holding(dic_mrg_operate, rdq_operate, symbols_buy, 'buy')
    pd_mrg_operate = pd.DataFrame(dic_mrg_operate)

    pd_holding = pd.DataFrame({'symbol': ['free_cash'], 'shares': [10000], 'rdq_0_1st': [None],
                               'rdq_0': [None], 'rdq_pq4': [None], 'pred': [None], 'num_p': [None], 'cost': [None],
                               'eval_metric_quantile': [None]})
    pd_fr_record = pd_fr_record_final.iloc[[0]]
    time_start = time.time()
    for i in range(len(pd_mrg_operate)):
        symbol, rdq_operate, operate_type = pd_mrg_operate.iloc[i][['symbol', 'rdq', 'action']]
        if operate_type == 'buy':
            dict_eval = {'symbol': [symbol], 'rdq_0': [rdq_operate], 'rdq_pq4': [rdq_operate], transaction.eval_metric: [1], 'num_p': [1]}
            pd_entry = pd.DataFrame(dict_eval).iloc[0]
            pd_entry.eval_metric_quantile = 1
            rdq_b = rdq_operate
            pd_fr_record, pd_holding = transaction.buy_share(pd_fr_record, pd_holding, pd_entry, rdq_b, operate_type)
        elif operate_type == 'sell':
            pd_fr_record, pd_holding = transaction.sell_share(pd_fr_record, pd_holding, symbol, rdq_operate, operate_type)
        time_span = round(time.time() - time_start, 1)
        print(f'\rTime: {time_span} s - progress {i + 1}/{len(pd_mrg_operate)}', end='')
    print()
    transaction.set_free_n_stock(False)

    time_start, time_end = dict_transform['decision_time_start'], str(datetime.datetime.now())[:10]
    pd_holding_final = get_holding_value(pd_holding, decision_time_end=time_end).copy()


    n_years = ((pd.to_datetime(time_end) - pd.to_datetime(time_start)).days) / 365
    value_final_mrg = round(pd_holding_final.value.sum())
    growth_rate = round((10 ** (np.log10(value_final_mrg / 10000) / n_years) - 1) * 100, 2)
    print(f"MRG overall growth rate: {growth_rate}% - value total {value_final_mrg}")

if 'plot distribution' == 'a':
    n_shift = 7
    files = [i for i in glob.glob('result/dict_save_data_*.pkl') if 'modified' not in i]
    versions = [int(i[-6:-4]) for i in files]
    pd_files = pd.DataFrame({'path': files, 'version': versions})
    pd_files = pd_files.loc[pd_files.version >= n_shift]

    fig, ax = plt.subplots(4, 2, figsize=(8, 9))
    ax = fig.axes

    for i in range(len(pd_files)):
        dict_filename = pd_files.iloc[i]['path']
        with open(dict_filename, 'rb') as handle:
            dict_data = pickle.load(handle)
        pd_value = dict_data['pd_value']
        ax[i].hist(pd_value.annual_growth, bins=10)
        ax[i].set_xlim(0.16, 0.30)

    fig.tight_layout()