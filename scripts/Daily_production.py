__author__ = 'Yunsong Xie'
__email__ = 'xiefinance00@gmail.com'
__company__ = 'Xie Finance LLC'

import os, datetime, sqlite3, time, glob, pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import lib as common_func
from sklearn.ensemble import RandomForestRegressor
import xgboost
import multiprocessing as mp

pd.set_option('display.max_column', 75)
pd.set_option('display.max_colwidth', 2400)
pd.set_option('display.width', 25000)

DIR = common_func.misc.get_main_dir()
stock_price = common_func.StockPrice()
stock_earn = common_func.StockEarning()

dir_fr = f'{DIR}\\static\\Financial_reports\\Wharton\\'
path_fr_db = f'{dir_fr}/fr_wharton.db'

try:
    _ = pd.read_sql("select * from col_name limit 10", con)
except:
    con = sqlite3.connect(path_fr_db)

if __name__ == '__main__':
    data_earn = stock_earn.get_yf_earning_calendar(date_start=common_func.date(-30), date_end=common_func.date(7))
