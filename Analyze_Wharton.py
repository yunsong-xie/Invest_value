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

import lib as common_func

pd.set_option('display.max_column', 60)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 10000)

DIR = os.path.dirname(os.path.abspath(__file__))

stock_price = common_func.StockPrice()

dir_fr = f'{DIR}\\static\\Financial_reports\\Wharton\\'
path_fr = max(glob.glob(f'{dir_fr}/wharton_FR_*.pkl'))


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
    _ = pd_wharton_fr.iloc[0]
except:
    pd_wharton_fr = pd.read_pickle(path_fr)
path_columns = glob.glob(f'{dir_fr}/wharton_columns.csv')[0]
pd_columns = pd.read_csv(path_columns)
pd_columns = pd_columns.sort_values(by='Variable Name')

desc_list = list(pd_columns['Description'])
desc_list = [' ('.join(i.split(' -- ')[-1].split(' (')[:-1]) for i in desc_list]
pd_columns['desc'] = desc_list
dict_columns = pd_columns.set_index('Variable Name')['desc'].to_dict()
dict_columns = {i.lower(): dict_columns[i] for i in dict_columns}
symbols = sorted(pd_wharton_fr['tic'].unique())


pd_view = get_pd_view(pd_wharton_fr.loc[(pd_wharton_fr.tic == 'CDNA') & (pd_wharton_fr.datafqtr >= '2019')],
                      dict_columns)

keyword_study = 'date'
cols_select_study = [i for i in dict_columns if ((keyword_study in dict_columns[i].lower()) &
                                                 (i in list(pd_wharton_fr.columns)))]
pd_temp = pd_wharton_fr[cols_select_study].drop_duplicates()
index_temp = pd_temp.index
pd_temp_1 = pd_wharton_fr.loc[index_temp][['tic'] + cols_select_study]
pd_view_1 = get_pd_view(pd_temp_1, dict_columns)


if 1 == 0:
    stock_price.update_price_symbol(symbols, force_reload=False)


for symbol in symbols:
    _ipo_date = stock_price.pd_listing.loc[stock_price.pd_listing.symbol == symbol].ipoDate.iloc[0].replace('-', '')
    pd_fr_symbol = pd_wharton_fr.loc[pd_wharton_fr.tic == symbol]
    pd_fr_symbol = pd_fr_symbol.loc[pd_fr_symbol['rdq'] >= _ipo_date]
    pd_fr_symbol['rdq']