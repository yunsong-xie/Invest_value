__author__ = 'Yunsong Xie'
__email__ = 'xiefinance00@gmail.com'
__company__ = 'Xie Finance LLC'


import time, os, glob
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select

import numpy as np
import pandas as pd
import lib as common_func

DIR = os.path.dirname(os.path.realpath(__file__))

pd.set_option('display.max_column', 15)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)


if __name__ == '__main__0':

    # pd_table_info_all_ori = pd.read_pickle('pd_table_info_all.pkl')
    dir_download = "D:/PycharmProjects/Investment/Invest_value/capitalIQ/static/Financial_reports/SEC"
    symbols = list(pd.read_csv(f'{DIR}/static/Financial_reports/pd_symbol.csv').symbol)
    symbols_error = list(pd.read_csv(f'{DIR}/static/Financial_reports/pd_symbol_error.csv').symbol)
    symbols_completed = [os.path.basename(i)[:-4] for i in glob.glob(f'{dir_download}/*.csv')]
    symbols = [i for i in symbols if i not in set(symbols_completed)]
    symbols = [i for i in symbols if i not in set(symbols_error)]
    symbols_exclude = ['CATM']
    symbols = [i for i in symbols if i not in set(symbols_exclude)]

    pd_table_info_list = []
    default_url = 'https://www.sec.gov/edgar/browse/?CIK={}&owner=exclude'
    driver_path= f'{DIR}/static/analysis/chromedriver.exe'

    chromeOptions = webdriver.ChromeOptions()
    prefs = {"download.default_directory": dir_download}
    chromeOptions.add_experimental_option("prefs", prefs)

    self = common_func.StockPrice()
    driver = webdriver.Chrome(executable_path=driver_path, options=chromeOptions)

    time_start = time.time()
    filename_default = f'{dir_download}/EDGAR Entity Landing Page.csv'
    for i_symbol, symbol in zip(range(len(symbols)), symbols):
        for i_trial in range(5):
            try:
                cik = self.get_cik(symbol)
                if cik is None:
                    continue
                cik = cik.lstrip('0')
                driver.get(default_url.format(cik))

                common_func.web_wait_element(driver, 'id', 'btnViewAllFilings', time_out=5)
                count_wait = 0
                while count_wait <= 10:
                    try:
                        view_button = driver.find_element_by_id('btnViewAllFilings')
                        if len(view_button.text) > 3:
                            if view_button.text == 'View filings':
                                view_button.click()
                                break
                            elif view_button.text == 'Hide filings':
                                break
                    except Exception as _:
                        print(_)
                    time.sleep(0.5)
                    count_wait += 1

                common_func.web_wait_element(driver, 'id', 'btnGroupDrop1', time_out=5)
                count_wait = 0
                while count_wait <= 6:
                    try:
                        open_view_group = driver.find_element_by_id('btnGroupDrop1')
                        open_view_group.click()
                        break
                    except:
                        time.sleep(0.5)
                    count_wait += 1


                view_group = driver.find_element_by_id('searchForms')
                try:
                    view_group.find_element_by_partial_link_text('Annual & quarterly reports').click()
                except:
                    a = 1

                common_func.web_wait_element(driver, 'id', 'filingDateFrom', time_out=5)
                count_wait = 0
                while count_wait <= 6:
                    try:
                        date_from = driver.find_element_by_id('filingDateFrom')
                        date_from.clear()
                        date_from.send_keys('1995-01-01')
                        break
                    except:
                        time.sleep(0.5)
                    count_wait += 1

                common_func.web_wait_element(driver, 'id', 'filingDateFrom', time_out=5)
                count_wait = 0
                while count_wait <= 6:
                    try:
                        main_table = driver.find_element_by_id('filingsTable_wrapper')
                        button_all = main_table.find_element_by_class_name('dt-buttons')
                        button_csv = button_all.find_elements_by_css_selector("button")[1]
                        time.sleep(0.75)
                        button_csv.click()
                        break
                    except:
                        time.sleep(0.5)
                    count_wait += 1

                time.sleep(2)

                count_wait = 0
                filename_new = f'{dir_download}/{symbol}.csv'
                while (not os.path.isfile(filename_new)) & (count_wait < 3):
                    count_wait += 1
                    os.rename(filename_default, filename_new)
                    time.sleep(1)

                time_span = round(time.time() - time_start, 1)
                print(f'\rGetting fine price info - time {time_span} s - progress {i_symbol + 1}/{len(symbols)} - {symbol}', end='')
                break
            except Exception as _:
                print(f'\n{_}')


