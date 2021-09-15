__author__ = 'Yunsong Xie'
__email__ = 'xiefinance00@gmail.com'
__company__ = 'Xie Finance LLC'


import re, os, sys, datetime, pickle
import numpy as np
import pandas as pd
import time
import glob, xlrd
from matplotlib import pyplot as plt
import lib as common_func
import multiprocessing as mp
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

pd.set_option('display.max_column', 60)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 10000)

DIR = os.path.dirname(os.path.abspath(__file__))

dir_fr = f'{DIR}\\static\\Financial_reports\\Wharton\\'
path_csv = max(glob.glob(f'{dir_fr}/wharton_FR_*.csv'))
path_columns = glob.glob(f'{dir_fr}/wharton_columns.csv')[0]
pd_columns = pd.read_csv(path_columns)
pd_columns = pd_columns.sort_values(by='Variable Name')

desc_list = list(pd_columns['Description'])
desc_list = [' ('.join(i.split(' -- ')[-1].split(' (')[:-1]) for i in desc_list]
pd_columns['desc'] = desc_list
dict_columns = pd_columns.set_index('Variable Name')['desc'].to_dict()
dict_columns = {i.lower(): dict_columns[i] for i in dict_columns}
dict_types = pd_columns.set_index('Variable Name')['Type'].to_dict()
dict_types_date = {i.lower() for i in dict_types if dict_types[i] == 'date'}
dict_types = {i.lower(): dict_types[i] if dict_types[i] != 'date' else 'str' for i in dict_types}

pd_wharton = pd.read_csv(path_csv)

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
    # pd_view_cash = pd_view.loc[pd_view.col_name.str.lower().str.contains('cash')]
    return pd_view



columns = list(pd_wharton.columns)
columns_in = [i for i in columns if i in dict_columns]
columns_exclude = [i for i in columns if i not in dict_columns]
pd_wharton_in = pd_wharton[columns_in]
pd_wharton_out = pd_wharton[columns_exclude]

temp = [i for i in dict_types if dict_types[i] == 'date']
pd_wharton_in = pd_wharton_in.astype(dict_types)
for key in dict_types_date:
    pd_wharton_in[key] = pd_wharton_in[key].str[:8]


pd_exclude_keywords = pd.read_csv(f'{DIR}/static/csv/wharton/exclude_keywords.csv')
columns_exclude_keywords = list(pd_exclude_keywords.keyword)

cols_select = list(dict_columns)
for keyword in columns_exclude_keywords:
    cols_select = [i for i in cols_select if keyword not in dict_columns[i].lower()]

pd_wharton_in = pd_wharton_in[cols_select]

pd_view = get_pd_view(pd_wharton_in.loc[(pd_wharton_in.tic == 'CDNA') &(pd_wharton_in.datafqtr >= '2019')],
                      dict_columns)


keyword_study = 'fof'
cols_select_study = [i for i in cols_select if keyword_study in dict_columns[i].lower()]
pd_temp = pd_wharton_in[cols_select_study].drop_duplicates()
index_temp = pd_temp.index
pd_temp_1 = pd_wharton_in.loc[index_temp][['tic'] + cols_select_study]
pd_view_1 = get_pd_view(pd_temp_1, dict_columns)
#pd_temp_2 = pd_temp_1.groupby('tic').size().reset_index()

date_now = str(datetime.datetime.now())[:10].replace('-', '')
# pd_wharton_in.to_pickle(f'{dir_fr}/wharton_FR_{date_now}.pkl')