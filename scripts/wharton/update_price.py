__author__ = 'Yunsong Xie'
__email__ = 'xiefinance00@gmail.com'
__company__ = 'Xie Finance LLC'

import sqlite3
import numpy as np
import pandas as pd
import glob

import lib as common_func

pd.set_option('display.max_column', 60)
pd.set_option('display.max_colwidth', 1200)
pd.set_option('display.width', 12000)


DIR = common_func.misc.get_main_dir()
stock_price = common_func.StockPrice()

dir_fr = rf'{DIR}\static\Financial_reports\Wharton\\'
path_fr_db = f'{dir_fr}/fr_wharton.db'


try:
    _ = pd.read_sql("select * from col_name limit 10", con)
except:
    con = sqlite3.connect(path_fr_db)


if __name__ == '__main__':
    pd_symbols = pd.read_sql("select distinct tic from report", con)
    symbols = sorted(pd_symbols['tic'])
    self = stock_price
    self.update_price_symbol(symbols, force_reload=False, check_abnormal=False)