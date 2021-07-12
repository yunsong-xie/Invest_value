
import re, os, sys, traceback
import numpy as np
import pandas as pd
import datetime
from bs4 import BeautifulSoup
import time
import urllib.request
from termcolor import colored

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service

DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(DIR).replace('\\', '/')+'/Alpha_Vantage')

import AV_query
global service, driver, dict_driver

from selenium.webdriver.chrome.options import Options
chrome_options = Options()
chrome_options.add_argument("--headless")



pd.set_option('display.max_column', 15)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)


def initiate_driver():
    default_url='https://libguides.gatech.edu/go.php?c=22546655'
    def capital_iq_login(default_url):
        driver.get(default_url)
        login_data = {'username': 'Ru.Xie.21@mba.mccombs.utexas.edu',
                      'password': 'Tuesday2010',
                      }
        for i in login_data:
            driver.find_element_by_id(i).send_keys(login_data[i])
        driver.find_element_by_id('myLoginButton').click()
        #Wait for the viewall button to show up
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "ctl03__rangeSlider_viewAll")))
        try:
            WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, "readOnlyDialog")))
            read_only_window = driver.find_element_by_id('readOnlyDialog')
            button_OK = read_only_window.find_element_by_class_name('close')
            button_OK.click()
        except Exception as _:
            print('No read only window shown')

        try:
            WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, "onetrust-accept-btn-handler")))
            button_cookies = driver.find_element_by_id('onetrust-accept-btn-handler')
            button_cookies.click()
        except Exception as _:
            print('No accept cookies button')


    driver_path= '../static/analysis/msedgedriver.exe'
    service = Service(driver_path)
    service.start()
    driver = webdriver.Remote(service.service_url)
    capital_iq_login(default_url)
    dict_driver={'service': service, 'driver': driver}
    return dict_driver


def web_wait_element(driver, element_id, time_out=5):
    time_delay=0.15
    time_start=time.time()
    while time.time()-time_start<=time_out:
        try:
            found_item=driver.find_element_by_id(element_id)
            return found_item
        except:
            time.sleep(time_delay)
    return None


def go_to_capital_statement_page(dict_driver, dict_stock_info, init_page=1, count_trial=0, random_wait=2):
    service, driver = dict_driver['service'], dict_driver['driver']
    time_start = time.time()
    time_delay, time_out, n_trials=0.15, 3, 3
    count_trial+=1
    exchange_ticker=dict_stock_info['exchange_ticker']
    if count_trial>n_trials:
        print(colored('\nWarning: Sometime went wrong during collect information for '+exchange_ticker+
                      '. Failed after '+str(n_trials)+' .', 'yellow'))
        return dict_driver, 'Failed', 'Failed'
    def go_search_symbol(driver, dict_stock_info):
        if 'company_id' in dict_stock_info:
            try:
                temp=int(dict_stock_info['company_id'])
                label_seach_for_company=0
            except:
                label_seach_for_company=1
        else:
            label_seach_for_company=1

        if label_seach_for_company==0:
            time.sleep(0.5)
            company_id=dict_stock_info['company_id']
            driver.get('https://www.capitaliq.com/CIQDotNet/Financial/Capitalization.aspx?CompanyId='+company_id)
        else:
            time.sleep(0.5)
            driver.find_element_by_id('SearchTopBar').send_keys('(' + exchange_ticker + ')')
            time.sleep(0.5)
            driver.find_element_by_id('SearchTopBar').send_keys(Keys.RETURN)


    def identify_current_page(driver, time_out):
        time_start=time.time()
        while time.time()-time_start<=time_out:
            try:
                element=driver.find_element_by_id('_pageHeader__pageTitle_PageHeaderLabel')
                if exchange_ticker.upper() in element.text.upper():
                    # right company, right page
                    data_info = driver.find_element_by_id('ctl03__tagLine').text
                    if 'of the trading currency' in data_info:
                        # right company, right page and data is available
                        return 'Capital_statement'
                    else:
                        # right company, right page, but data is not available, need to reload driver
                        return 'Data_no_available, need reload'
                else:
                    # not the right company
                    return 'Random'
            except:
                try:
                    element=driver.find_element_by_id('CompanyHeaderInfo')
                    if exchange_ticker.upper() in element.text.upper():
                        # Right company, not the right page
                        if 'PUBLIC FUND' in element.text.upper():
                            # Its a public fund, no finanical information
                            return 'Public_fund'
                        else:
                            return 'Company_default'
                    else:
                        # not the right company
                        return 'Random'
                except:
                    try:
                        # The company has duplicated entries and go to search page
                        # If so, search the exchange_ticker in the found table and click on it, return Random
                        driver.find_element_by_id('Header_PageHeaderLabel')
                        element = driver.find_element_by_id('ctl01__SearchGridView')
                        element.find_element_by_partial_link_text('(' + exchange_ticker + ')').click()
                        return 'Potential_Company_default'
                    except:
                        try:
                            driver.find_element_by_id('_profileHelp_Displaysection1__suggestedResults__searchPrompt')
                            element = driver.find_element_by_class_name('suggestBox')
                            element.find_element_by_partial_link_text(exchange_ticker).click()
                            return 'No_company_found_return_suggested_page'
                        except:
                            time.sleep(time_delay)
        # Time out and not the right company, not the right page
        return 'Random'

    if init_page==1:
        # re-search this symbol
        go_search_symbol(driver, dict_stock_info)
    elif init_page >= 100:
        # Global initiate the driver
        dict_driver['service'].stop()
        dict_driver = initiate_driver()
        service, driver = dict_driver['service'], dict_driver['driver']
        print('Problem accurs')
    time.sleep(0.5)
    page_status=identify_current_page(driver, time_out)
    company_type='company'
    if page_status=='Capital_statement':
        company_id=re.findall('COMPANYID=([0-9]*)', driver.current_url.upper())[0]
    elif page_status=='Company_default':
        # if this is the company default page, input capital statement page and rerun this function
        company_id=re.findall('COMPANYID=([0-9]*)', driver.current_url.upper())[0]
        capital_statement_url='https://www.capitaliq.com/CIQDotNet/Financial/Capitalization.aspx?CompanyId='+company_id
        driver.get(capital_statement_url)
        time.sleep(np.random.random()*random_wait)
        dict_driver, company_id, company_type = go_to_capital_statement_page(dict_driver, dict_stock_info,
                                                                             init_page=0, count_trial=count_trial)
    elif page_status=='Public_fund':
        company_id = re.findall('COMPANYID=([0-9]*)', driver.current_url.upper())[0]
        company_type='fund'
    elif page_status == 'Potential_Company_default':
        # redirect from duplicated found entries, similar to Random, but do not need to reinitiate
        time.sleep(np.random.random() * random_wait)
        dict_driver, company_id, company_type = go_to_capital_statement_page(dict_driver, dict_stock_info,
                                                                             init_page=0, count_trial=count_trial)
    elif page_status == 'No_company_found_return_suggested_page':
        # redirect from suggested company, similar to Random, but do not need to reinitiate
        time.sleep(np.random.random() * random_wait)
        dict_driver, company_id, company_type = go_to_capital_statement_page(dict_driver, dict_stock_info,
                                                                             init_page=0, count_trial=count_trial)
    elif page_status == 'Data_no_available, need reload':
        # Data_no_available, need reload driver
        time.sleep(np.random.random() * random_wait)
        dict_driver, company_id, company_type = go_to_capital_statement_page(dict_driver, dict_stock_info,
                                                                             init_page=init_page+50, count_trial=count_trial)
    else:
        # if this is the company default page, input capital statement page and rerun this function
        time.sleep(np.random.random() * random_wait)
        dict_driver, company_id, company_type = go_to_capital_statement_page(dict_driver, dict_stock_info,
                                                                             init_page=1, count_trial=count_trial)

    return dict_driver, company_id, company_type


def get_queried_info(pd_exchange):
    path_financial_xls_folder=r'C:\Users\yunso\Downloads'
    financial_xls_list=[i.upper() for i in os.listdir(path_financial_xls_folder) if i[-4:]=='.xls']
    found_list=[]
    for i in range(len(pd_exchange)):
        exchange_tickler=pd_exchange.iloc[i].exchange_ticker.upper()
        exchange_tickler_components=exchange_tickler.split(':')

        label_found = 0
        for financial_xls in financial_xls_list:
            financial_xls_components=financial_xls.split(' ')
            if ((exchange_tickler_components[0] in financial_xls_components) &
                (exchange_tickler_components[1] in financial_xls_components)):
                label_found=1
                break
        found_list.append(label_found)
    pd_exchange['label_found']=found_list
    return pd_exchange


def get_exist_xls(exchange_ticker):
    folder = 'C:/Users/yunso/Downloads'
    files = os.listdir(folder)
    info_list = exchange_ticker.split(':')
    for file in files:
        if (info_list[0] in file) & (info_list[1] in file):
            return True
    return False


if __name__ == '__main__':
    dict_driver = initiate_driver()
    service, driver = dict_driver['service'], dict_driver['driver']
    av_session = AV_query.financial_data()
    pd_ciq = av_session.pd_ciq

    symbols = list(pd_ciq.symbol)
    dict_ciq_id = {'symbol': [], 'exchange_ticker': [], 'company_id': [], 'company_type': []}
    count_process = 0
    time_start = time.time()

    def section(dict_driver, dict_stock_info):
        dict_driver, company_id, company_type = go_to_capital_statement_page(dict_driver, dict_stock_info, init_page=1,
                                                                             count_trial=0, random_wait=2.5)
        service, driver = dict_driver['service'], dict_driver['driver']
        if company_type == 'company':
            try:
                web_wait_element(driver, element_id='ctl03__rangeSlider_viewAll', time_out=5)
                driver.find_element_by_id('ctl03__rangeSlider_viewAll').click()
                web_wait_element(driver, element_id='ctl03__rangeSlider_viewAll', time_out=5)
                time.sleep(np.random.random() * 2.5)
                _ = driver.find_element_by_id('_pageHeader__excelReport').click()
                time.sleep(np.random.random() * 2.5)
            except:
                print(colored(
                    '\nWarning: not able to parse symbol: ' + dict_stock_info['symbol'] +
                    '. It has only limited period of financial reports.',
                    'yellow'))

    for symbol in symbols:
        count_process += 1
        pd_exchange_entry = pd_ciq.loc[pd_ciq.symbol == symbol].iloc[0]
        exchange_ticker = pd_exchange_entry.exchange_ticker
        dict_stock_info = {'symbol': symbol, 'exchange_ticker': exchange_ticker}

        time_span = str(round(time.time() - time_start, 1))
        print('\rCurrent time: ' + str(datetime.datetime.now())[:19] +
              ' - Time: ' + time_span + ' s. Current progress: ' + exchange_ticker + ' - ' + str(
            count_process) + '/' + str(len(symbols)), end='')


        if not get_exist_xls(exchange_ticker):
            _success, _reset = False, False
            while not _success:
                try:
                    if _reset:
                        dict_driver = initiate_driver()
                        service, driver = dict_driver['service'], dict_driver['driver']
                    section(dict_driver, dict_stock_info)
                    _success = True
                except Exception as _:
                    error_msg = traceback.format_exc()
                    print(error_msg)
                    print(f'Fatal Error happend for {symbol}')
                    _reset = True
                    time.sleep(15 + np.random.random() * 15)
                    try:
                        dict_driver['service'].stop()
                    except:
                        a = 1
                time.sleep(2 + np.random.random() * 8)

