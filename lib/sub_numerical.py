# Author: Yunsong Xie
import numpy as np
import pandas as pd
from scipy.stats import norm

def clean_outliers(pd_data_input, col_list, sigma=3, ignore_discrete=True, n_discrete=3,
                   replace=False, stats_output=True):
    """
    clean the outlier with the assigned sigma threshold (default: 5)
    It uses a while funtion, so that the it keeps cleaning the outliers until there is none
    :param pd_data_input:
    :param col_list:
    :param sigma:
    :param ignore_discrete:
    :param n_discrete:
    :param replace:
    :return:
    """

    if replace == 1:
        pd_data = pd_data_input.copy()
    else:
        pd_data = pd_data_input

    # if col_list is str, makes it list
    if type(col_list) is str:
        col_list = [col_list]

    # if the column is discrete number, ignore
    if ignore_discrete == 1:
        for out_lier_col in col_list:
            if len(pd_data[out_lier_col].unique()) <= n_discrete:
                col_list.remove(out_lier_col)

    if len(col_list) > 0:
        n_old = len(pd_data)
        label_continue = True
        for col in col_list:
            pd_data = pd_data.loc[np.isfinite(pd_data[col])]

        count = 0
        dict_stats = {}
        while label_continue:
            count += 1
            label_continue = False
            for col in col_list:

                value_array = pd_data[col]
                mean, std = norm.fit(value_array)

                if not replace:
                    pd_data = pd_data.loc[(pd_data[col] <= (mean + std * sigma)) & (
                                pd_data[col] >= (mean - std * sigma))]

                else:
                    if ignore_discrete == 1:
                        if len(pd_data[col].unique()) <= n_discrete:
                            continue

                    pd_data.loc[pd_data[col] < (mean - std * sigma), col] = (mean - std * sigma)
                    pd_data.loc[pd_data[col] > (mean + std * sigma), col] = (mean + std * sigma)

                    dict_stats[col] = {'type': 'clean', 'mean': mean, 'std': std, 'sigma': sigma}

                    value_array = pd_data[col]
                    mean_eval, std_eval = norm.fit(value_array)

                    if ~(((mean_eval + std_eval * sigma) >= (mean + std * sigma)) |
                         ((mean_eval - std_eval * sigma) <= ((mean - std * sigma)))):
                        if not stats_output:
                            label_continue = True

                    # print('\r',count, mean_eval+std_eval*sigma, (mean+std*sigma), end='')


            if not replace:
                n_new = len(pd_data)
                if n_new < n_old:
                    label_continue = True
                    n_old = len(pd_data)
                else:
                    label_continue = False
    if replace:
        return pd_data, dict_stats
    else:
        return pd_data