# Author: Yunsong Xie
import re, os, time, glob
import datetime
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from termcolor import colored
from zipfile import ZipFile
import io

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select

pd.set_option('display.max_column', 15)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)
DIR = os.path.dirname(os.path.abspath(__file__))

def make_soup(url):
    thepage=urllib.request.urlopen(url)
    soupdata=BeautifulSoup(thepage,"html.parser")
    return soupdata



def web_wait_element(driver, element_type='id', element_value='', time_out=5):
    time_delay, time_start = 0.15, time.time()
    while (time.time() - time_start) <= time_out:
        if element_type == 'id':
            found_items = driver.find_elements_by_id(element_value)
        elif element_type == 'class':
            found_items = driver.find_elements_by_class_name(element_value)
        elif element_type == 'css':
            found_items = driver.find_elements_by_css_selector(element_value)
        elif element_type == 'text':
            found_items = driver.find_element_by_partial_link_text(element_value)
        else:
            raise ValueError("Can't identify the element type")
        if len(found_items) > 0:
            return found_items[0]
        time.sleep(time_delay)


def date(days=0):
    date_now = datetime.datetime.now()
    date_delta = datetime.timedelta(days=days)
    date_output = date_now + date_delta
    return str(date_output)[:10]


def unix2date(unix_time):
    # datetime_output = datetime.datetime.utcfromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')
    datetime_output = datetime.datetime.fromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')
    return datetime_output


def date2unix(time_date):
    time_date_parse = datetime.datetime.strptime(time_date, '%Y-%m-%d')
    timestamp = int(datetime.datetime.timestamp(time_date_parse))
    return timestamp


def get_timestamp_now():
    now = datetime.datetime.now()
    timestamp_now = int(datetime.datetime.timestamp(now))
    return timestamp_now


def parse_yahoo_data(soup):
    # this function takes the raw input from the yahoo finance
    # parse it and output a 2D numpy data that has 6 columes
    # TIME, CLOSE, HIGH, LOW, OPEN, VOLUME
    # Time is expressed as year/month/day/sequance, 9 digits in total

    # start and end strings for time data bulk
    soup_str = str(soup)

    query_type = re.findall('\"dataGranularity\":\"([\w]+)\"', soup_str)[0]
    if query_type == '1d':
        headers = ['timestamp', 'open', 'high', 'low', 'volume', 'close', 'adjclose']
    else:
        headers = ['timestamp', 'open', 'high', 'low', 'volume', 'close']


    def parse_info(header):
        info_chunk = re.findall(f'"{header}":\[([E\-\d,\.null]*)', soup_str)
        if len(info_chunk)>1:
            info_chunk = sorted(info_chunk, key=lambda x: len(x), reverse=True)[:-1]
        info_list_output = info_chunk[0].split(',')
        return info_list_output

    dict_sum = {}
    for header in headers:
        info_list = parse_info(header)
        dict_sum[header] = info_list

    pd_result = pd.DataFrame(dict_sum)
    pd_result = pd_result.loc[pd_result.close != 'null']
    for header in headers:
        info_list = list(pd_result[header])
        if header == 'timestamp':
            info_list = [unix2date(int(i)) for i in info_list]
        elif header == 'volume':
            info_list = [int(i) for i in info_list]
        else:
            info_list = [round(float(i), 4) for i in info_list]
        pd_result[header] = info_list
        # print(header, len(info_list))

    pd_result = pd_result.rename(columns={'timestamp': 'time'})

    return pd_result


def get_pd_exchange(exchange_path, filename_full, min_marketcap = 0.75):
    pd_exchange = pd.read_csv(filename_full, sep=",")
    keys = list(pd_exchange.keys())[:-2]

    pd_exchange = pd_exchange[keys]
    dict_rename = {i: i.lower() for i in keys}
    pd_exchange = pd_exchange.rename(columns=dict_rename)
    pd_exchange = pd_exchange.loc[~pd_exchange.marketcap.isna()].copy()

    pd_exchange.marketcap = pd_exchange.marketcap.str.replace('$', '').str.upper()
    pd_exchange['scale'] = 0
    pd_exchange.loc[pd_exchange.marketcap.str.contains('B'), 'scale'] = 1
    pd_exchange.loc[pd_exchange.marketcap.str.contains('M'), 'scale'] = 0.001

    pd_exchange.marketcap = pd_exchange.marketcap.str.strip('BM').astype(float)
    pd_exchange.marketcap = pd_exchange.marketcap * pd_exchange.scale
    pd_exchange = pd_exchange[pd_exchange.keys()[:-1]]

    pd_exchange = pd_exchange.loc[pd_exchange.marketcap >= min_marketcap]

    pd_exchange_ticker = pd.read_csv(exchange_path + '/' + 'Capital_ID_exchange_ticker.csv')
    pd_exchange_ticker['symbol'] = pd_exchange_ticker.exchange_ticker.str.split(':').str[-1]
    pd_exchange = pd_exchange.merge(pd_exchange_ticker, left_on='symbol', right_on='symbol', how='inner')

    pd_exchange['exchange'] = pd_exchange.exchange_ticker.str.split(':').str[0]
    pd_exchange = pd_exchange.loc[pd_exchange.exchange.isin(['NYSE', 'NasdaqCM', 'NasdaqGM', 'NasdaqGS'])]

    return pd_exchange


def initiate_stock_symbols():
    """
    This function parse the url from noldnasdaq.com get the lastest stock symbol list
    The saved dir is at './assets/csv'
    Final csv is renamed to be 'nasdaq.csv' and 'nyse.csv'
    :param dict_url:
    :return:
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options

    dict_url = {'nasdaq': 'https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download',
                'nyse': 'https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nyse&render=download'}
    path_temp = os.path.dirname(__file__)
    path_save = f"{path_temp}/assets/csv".replace('/','\\')
    exchange_list = list(dict_url.keys())
    time_delay = 0.1
    label_month = str(datetime.datetime.now()).replace('-', '')[:6]

    csv_list_exist = []
    for filename in exchange_list:
        path_file = f'{path_save}/{filename}_{label_month}.csv'
        if os.path.isfile(path_file):
            csv_list_exist.append(filename)
    download = not (len(csv_list_exist) == len(dict_url))

    if download:
        # Delete 'nasdaq.csv' and 'nyse.csv'
        for filename in exchange_list:
            path_file = f'{path_save}/{filename}_{label_month}.csv'
            if os.path.isfile(path_file):
                os.remove(path_file)
        csv_list_ori = [i for i in os.listdir(path_save) if i[-4:] == '.csv']

        if not os.path.isdir(path_save):
            os.makedirs(path_save)
        options = Options()
        options.add_experimental_option("prefs", {"download.directory_upgrade": True,
                                                  "safebrowsing.enabled": True,
                                                  "download.default_directory": path_save})

        def decide_finish(dict_url, csv_list_ori, path_save):
            csv_list = [i for i in os.listdir(path_save) if i[-4:] == '.csv']
            csv_diff_list = [i for i in csv_list if not i in csv_list_ori]
            label_boolean = len(csv_diff_list) != len(dict_url)
            return label_boolean, csv_diff_list

        label_boolean, csv_diff_list = decide_finish(dict_url, csv_list_ori, path_save)

        # Use chromedriver
        driver_path = os.path.dirname(__file__)+r'/assets/chromedriver.exe'
        service = Service(driver_path)
        service.start()
        driver = webdriver.Remote(service.service_url, options=options)

        count = 0
        for filename in exchange_list:
            count += 1
            url = dict_url[filename]
            driver.switch_to.window(driver.window_handles[-1])
            driver.execute_script(f'''window.open("{url}","_blank");''')
            driver.switch_to.window(driver.window_handles[-1])

            while True:
                try:
                    download = driver.find_element_by_class_name('nasdaq-screener__download')
                    download.click()
                    break
                except:
                    a = 1
                    time.sleep(time_delay)

            while count != len(csv_diff_list):
                time.sleep(time_delay)
                label_boolean, csv_diff_list = decide_finish(dict_url, csv_list_ori, path_save)

        service.stop()

        # rename files
        csv_diff_list = sorted(csv_diff_list)
        dict_file = {}
        for i in range(len(csv_diff_list)):
            csv_file, exchange = csv_diff_list[i], exchange_list[i]
            os.rename(f'{path_save}\\{csv_file}', f'{path_save}\\{exchange}_{label_month}.csv')
            dict_file[exchange] = f'{path_save}\\{exchange}_{label_month}.csv'
    else:
        dict_file = {}
        for i in range(len(exchange_list)):
            exchange = exchange_list[i]
            dict_file[exchange] = f'{path_save}\\{exchange}_{label_month}.csv'

    return dict_file


def parse_stock_file(dict_exchange, marketcap_filter):
    pd_exchange_list = []
    for key in dict_exchange:
        path_file = dict_exchange[key]
        pd_exchange = pd.read_csv(path_file)

        keys = list(pd_exchange.keys())[:-2]

        pd_exchange = pd_exchange[keys]
        dict_rename = {i: i.lower() for i in keys}
        dict_rename['Market Cap'] = 'marketcap'
        pd_exchange = pd_exchange.rename(columns=dict_rename)
        pd_exchange = pd_exchange.loc[~pd_exchange.marketcap.isna()].copy()

        if str(pd_exchange['marketcap'].dtypes) != 'float64':
            pd_exchange.marketcap = pd_exchange.marketcap.str.replace('$', '').str.upper()
            pd_exchange['scale'] = 0
            pd_exchange.loc[pd_exchange.marketcap.str.contains('B'), 'scale'] = 1
            pd_exchange.loc[pd_exchange.marketcap.str.contains('M'), 'scale'] = 0.001

            pd_exchange.marketcap = pd_exchange.marketcap.str.strip('BM').astype(float)
            pd_exchange.marketcap = pd_exchange.marketcap * pd_exchange.scale
            pd_exchange = pd_exchange[pd_exchange.keys()[:-1]]
        else:
            pd_exchange.marketcap = pd_exchange.marketcap / 10**9

        pd_exchange = pd_exchange.loc[pd_exchange.marketcap > marketcap_filter]
        pd_exchange_list.append(pd_exchange)
    pd_exchange_all = pd.concat(pd_exchange_list)
    pd_exchange_all = pd_exchange_all.sort_values(by='marketcap', ascending=False)
    pd_exchange_all.index = np.arange(len(pd_exchange_all))

    pd_exchange_all = pd_exchange_all.loc[pd_exchange_all.marketcap >= marketcap_filter]

    return pd_exchange_all


def get_symbol_list(marketcap=3.5):
    # date_now = str(datetime.datetime.now())[8:10]
    dict_exchange = initiate_stock_symbols()
    pd_exchange = parse_stock_file(dict_exchange, marketcap_filter=marketcap)
    return pd_exchange


class StockEarning:
    """
    Includes functions to pull earning dates and info
    """
    def __init__(self):
        self.pd_earning_date = None

    def get_yf_earning_dates(self, symbols):
        """
        Get earning table from yahoo finance page for selected symbols.
        Args:
            symbols (str/list): stock symbols to pull the earning dates info

        Returns:
            (pandas.dataframe): The earning dates info of selected symbol(s), columns include:
                symbol, time, eps_estimate, eps_reported, suprise
        """

        path_pickle_earning_dates = f'{os.path.dirname(DIR)}/static/Financial_reports/Earning_dates.pkl'
        dict_earning_dates = {'symbol': [], 'time': [], 'eps_estimate': [], 'eps_reported': [], 'suprise': []}

        if type(symbols) is str:
            symbols = [symbols]

        if self.pd_earning_date is None:
            if os.path.isfile(path_pickle_earning_dates):
                self.pd_earning_date = pd.read_pickle(path_pickle_earning_dates)
            else:
                self.pd_earning_date = pd.DataFrame(dict_earning_dates)
        symbols_crawl = []

        symbols_exist_set = set(self.pd_earning_date.symbol)
        for symbol in symbols:
            if symbol not in symbols_exist_set:
                symbols_crawl.append(symbol)

        if symbols_crawl:
            time_start = time.time()
            isSuccess, iscommit = True, False
            for i_symbol, symbol in zip(range(len(symbols_crawl)), symbols_crawl):
                count_trial = 0
                while (not isSuccess) | (count_trial == 0):
                    time_span = round(time.time() - time_start, 1)
                    count_trial += 1
                    try:
                        base_url = 'https://finance.yahoo.com/calendar/earnings?symbol='
                        url = f'{base_url}{symbol}'
                        soupdata = make_soup(url)
                        entry_class_name0 = 'simpTblRow Bgc($hoverBgColor):h BdB Bdbc($seperatorColor) Bdbc($tableBorderBlue):h H(32px) Bgc($lv1BgColor)'
                        entry_class_name1 = 'simpTblRow Bgc($hoverBgColor):h BdB Bdbc($seperatorColor) Bdbc($tableBorderBlue):h H(32px) Bgc($lv2BgColor)'
                        entrys = soupdata.find_all(attrs={'class': entry_class_name0})
                        entrys += soupdata.find_all(attrs={'class': entry_class_name1})
                        if len(entrys) == 0:
                            dict_earning_dates['symbol'].append(symbol)
                            dict_earning_dates['time'].append('NA')
                            dict_earning_dates['eps_estimate'].append(np.nan)
                            dict_earning_dates['eps_reported'].append(np.nan)
                            dict_earning_dates['suprise'].append(np.nan)
                        else:
                            for entry in entrys:
                                format_1 = '<span data-reactid="[0-9]+">([A-Za-z0-9, ]+)</span>'
                                format_2 = '<span class="Mstart\([0-5]+px\)" data-reactid="[0-9]+">([A-Z]+)</span>'
                                date_detects = re.findall(f'{format_1}[.\n]*{format_2}', str(entry))[0]
                                date_detect = str(datetime.datetime.strptime(date_detects[0][:12], '%b %d, %Y'))[:10]
                                eps_list = re.findall('-->([\-0-9\.]+)<!', str(entry))
                                surprise_list = re.findall('">([\+\-0-9\.]+)</span></td>', str(entry))
                                eps_list += surprise_list
                                eps_list = [float(i) if i != '-' else np.nan for i in eps_list]
                                dict_earning_dates['symbol'].append(symbol)
                                dict_earning_dates['time'].append(date_detect)
                                dict_earning_dates['eps_estimate'].append(eps_list[0])
                                dict_earning_dates['eps_reported'].append(eps_list[1])
                                dict_earning_dates['suprise'].append(eps_list[2])

                        isSuccess = True
                        print(f'\rTime: {time_span} - Progress {symbol}: {len(symbols_crawl)}/{i_symbol + 1}', end='')
                        time.sleep(2)

                    except Exception as _:
                        if isSuccess:
                            iscommit = True
                            isSuccess = False
                            print()
                        time.sleep(30)
                        print(colored(f'\rTime: {time_span} - problems in crawling {symbol} - '
                                      f'Trial {count_trial}: {_}', 'yellow'), end='')
                    if iscommit | (i_symbol % 20 == 0):
                        pd_earning_dats_symbols = pd.DataFrame(dict_earning_dates)
                        self.pd_earning_date = pd.concat([self.pd_earning_date, pd_earning_dats_symbols]).drop_duplicates()
                        self.pd_earning_date.to_pickle(path_pickle_earning_dates)

            if isSuccess:
                pd_earning_dats_symbols = pd.DataFrame(dict_earning_dates)
                self.pd_earning_date = pd.concat([self.pd_earning_date, pd_earning_dats_symbols]).drop_duplicates()
                self.pd_earning_date.to_pickle(path_pickle_earning_dates)
        pd_output = self.pd_earning_date.loc[self.pd_earning_date.symbol.isin(symbols)]
        return pd_output

    def get_earning_dates(self, symbols):
        return self.get_yf_earning_dates(symbols)

    def get_cik_data(self, force_reload=False):
        """
        Obtain cik vs symbol data from sec website
        Args:
            force_reload (Boolean): whether to force to reload

        Returns:
            (pandas.dataframe): corresponding table that includes both cik code and symbol
        """
        if self.pd_cik is None:
            filename_cik = f'{os.path.dirname(DIR)}/static/Financial_reports/sec_cik.csv'
            if (not os.path.isfile(filename_cik)) | (force_reload):
                url = 'https://www.sec.gov/files/company_tickers.json'
                pd_cik = pd.read_json(url).T
                pd_cik = pd_cik.rename(columns={'cik_str': 'cik', 'ticker': 'symbol'})
                pd_cik['cik'] = pd_cik['cik'].astype(str).str.rjust(10, '0')
                info_add = [['0001678211', 'CATM', 'CATM HOLDINGS LLC']]
                cik_set = set(pd_cik.cik)
                for i in range(len(info_add)):
                    cik = info_add[i][0]
                    if cik not in cik_set:
                        pd_add = pd.DataFrame({'cik': [info_add[i][0]], 'symbol': [info_add[i][1]], 'title': [info_add[i][2]]})
                        pd_cik = pd.concat([pd_cik, pd_add])
                pd_cik.to_csv(filename_cik, index=False)
            else:
                pd_cik = pd.read_csv(filename_cik)
                pd_cik['cik'] = pd_cik['cik'].astype(str).str.rjust(10, '0')
            self.pd_cik = pd_cik
            return pd_cik
        else:
            return self.pd_cik

    def get_cik(self, symbols):
        """
        Get the cik code for selected symbols
        Args:
            symbols (str/list): stock symbols to get cik code

        Returns:
            (str/dict): if input symbol is str, returns the cik code in str
                        if input symbol is list, returns the cik code in dict
                        all returned cik code are in str and with length of 10 with '0' in the front
        """
        if self.pd_cik is None:
            self.get_cik_data()
        dict_cik = self.pd_cik[['cik', 'symbol']].set_index('symbol').to_dict()['cik']
        if type(symbols) is str:
            symbol = symbols
            if symbol in dict_cik:
                return dict_cik[symbols]
            else:
                return None
        else:
            return {symbol: dict_cik[symbol] for symbol in symbols}

    def _obselete_get_10q(self, symbol):
        zip_file = f'{os.path.dirname(DIR)}/static/Financial_reports/SEC/companyfacts.zip'
        cik = self.get_cik(symbol)
        fzip = ZipFile(zip_file, 'r')
        freport = fzip.open(f'CIK{cik}.json')
        stringio = io.StringIO(freport.read().decode('utf-8'))
        pd_info_1 = pd.io.json.read_json(stringio)

        pd_info = pd_info_1.loc['us-gaap'].loc['facts']
        keys = list(pd_info.keys())

        pd_data_list = []
        data_sec_list = []
        for key in keys:
            for currency in pd_info[key]['units']:
                data_sec_list += [i for i in pd_info[key]['units'][currency]]
        pd_data = pd.DataFrame(data_sec_list).sort_values(by='end')
        pd_data.groupby('end').filed.min().reset_index()

    def get_sec_earning_data(self, symbols):
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
        driver_path = f'{DIR}/static/analysis/chromedriver.exe'

        chromeOptions = webdriver.ChromeOptions()
        prefs = {"download.default_directory": dir_download}
        chromeOptions.add_experimental_option("prefs", prefs)

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

                    web_wait_element(driver, 'id', 'btnViewAllFilings', time_out=5)
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

                    web_wait_element(driver, 'id', 'btnGroupDrop1', time_out=5)
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

                    web_wait_element(driver, 'id', 'filingDateFrom', time_out=5)
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

                    web_wait_element(driver, 'id', 'filingDateFrom', time_out=5)
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
                    print(f'\rGetting fine price info - time {time_span} s - progress {i_symbol + 1}/{len(symbols)} - {symbol}',
                          end='')
                    break
                except Exception as _:
                    print(f'\n{_}')

    def verify_earning_dates(self):
        a = 1




class StockPrice(StockEarning):
    """
    Includes functions to pull price as well as earning info
    """
    def __init__(self):
        super().__init__()
        self.dir_static = f'{os.path.dirname(DIR)}/static'
        self.dir_price = f'{self.dir_static}/pkl_price'
        self.pd_listing = None
        self.get_listing_info()

        self.dict_pd_price = {}
        self.pd_cik = None

    def get_listing_info(self):
        """
        Get the listing information. It caches the information by month. Every month it updates the info from alphavantage.

        Returns:
            (pandas.dataframe): listing information for  alpha_vantage list_status file
        """
        dir_pkl = f'{self.dir_static}/pkl_listing'
        path_listing = f'{dir_pkl}/listing_{date(0)[:7]}.pkl'
        if os.path.isfile(path_listing):
            self.pd_listing = pd.read_pickle(path_listing)
        else:
            self.pd_listing = pd.read_csv('https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=demo')
            self.pd_listing.to_pickle(path_listing)
        return self.pd_listing

    def get_price_range(self, symbol, date_start='1995-01-01', date_end=None, source='local', time_type='sec'):
        """
        Obtain intraday price info for selected symbol, this function only get the range defined by date_start, and date_end
        Args:
            symbol (str): Stock symbol
            date_start (str): Starting date
            date_end (str): end date (default: None)
            source (str): the source of this inquery, either online or local

        Returns:
            pandas.dataframe: The price info for the selected dates, in the case of the symbol does not exist,
                return None.
        """

        unix_start = date2unix(date_start)
        if not date_end:
            unix_end = get_timestamp_now()
        else:
            unix_end = date2unix(date_end)
        date_end = unix2date(unix_end)[:10]

        if source == 'online':
            url = f'https://query1.finance.yahoo.com/v7/finance/chart/' \
                  f'{symbol}?period1={unix_start}&period2={unix_end}&interval=1d'

            soup = make_soup(url)

            if len(str(soup)) > 770:
                pd_data = parse_yahoo_data(soup)
                pd_data['symbol'] = symbol
            else:
                keys = ['time', 'open', 'high', 'low', 'volume', 'close', 'adjclose', 'symbol']
                pd_data = pd.DataFrame({i:[] for i in keys})

            if time_type in ['day', 'date']:
                if str(pd_data.time.dtypes) == 'object':
                    pd_data.time = pd_data.time.str[:10]
        elif source == 'local':
            file_path = f'{self.dir_price}/{symbol}.pkl'
            if symbol in self.dict_pd_price:
                pd_data = self.dict_pd_price[symbol]
            else:
                if os.path.isfile(file_path):
                    pd_data = pd.read_pickle(file_path)
                    self.dict_pd_price[symbol] = pd_data
                else:
                    return None
            if time_type in ['day', 'date']:
                if str(pd_data.time.dtypes) == 'object':
                    pd_data.time = pd_data.time.str[:10]
            pd_data = pd_data.loc[(pd_data.time >= date_start) & (pd_data.time <= date_end)].copy()

        else:
            raise ValueError("Source has to be either 'online' or 'local'.")

        return pd_data

    def get_price_dates(self, symbol, dates, n_shift=0):
        """
        Obtain intraday price info for selected symbol, this function only get the info for the dates specified by "dates"
        Args:
            symbol: Stock symbol
            dates (str/list): dates for the info needed.
            n_shift: (int): days needs to be shifted from the requested dates, default: 0

        Returns:
            pandas.dataframe: The price info for the selected dates
        """
        if type(dates) is str:
            dates = [dates]
        if n_shift:
            dates = [unix2date(date2unix(i) + 3600 * 24 * n_shift)[:10] for i in dates]

        pd_data_output = self.get_price_range(symbol, date_start=date(-14), date_end=date(0))
        if len(pd_data_output) > 0:
            pd_data_2_list = []
            date_start_temp = unix2date(date2unix(min(dates)) - 3600 * 24 * 14)[:10]
            date_end_temp = unix2date(date2unix(max(dates)) + 3600 * 24 * 14)[:10]
            pd_data_date = self.get_price_range(symbol, date_start=date_start_temp, date_end=date_end_temp)
            pd_data_date['time'] = pd_data_date['time'].str[:10]
            pd_data_date_1 = pd_data_date.loc[pd_data_date.time.isin(dates)].copy()
            pd_data_date_1['time_request'] = pd_data_date_1['time']
            dates_miss = [i for i in dates if (i not in list(pd_data_date_1.time)) & (i <= max(pd_data_date.time)) &
                          (i >= min(pd_data_date.time))]
            if dates_miss:
                dates_miss_end = unix2date(date2unix(max(dates_miss)) + 3600 * 24 * 14)[:10]
                pd_data_date_2_ori = pd_data_date.loc[(pd_data_date.time > min(dates_miss)) & (pd_data_date.time <= dates_miss_end)]
                data_date = np.asarray(pd_data_date_2_ori.time)
                for _date in dates_miss:
                    pd_data_date_temp = pd_data_date_2_ori.loc[pd_data_date_2_ori.time>_date].iloc[[0]].copy()
                    pd_data_date_temp['time_request'] = _date
                    pd_data_2_list.append(pd_data_date_temp)
                pd_data_2_list.append(pd_data_date_1)
                pd_data_output = pd.concat(pd_data_2_list)
            else:
                pd_data_output = pd_data_date_1
        return pd_data_output

    def update_price_symbol(self, symbols, time_hard_start='1995-01-01', force_reload=False):
        """
        Update the local pkl stored historical intraday price information of provided list of symbols
        Args:
            symbols (str/list): Stock symbol of interested
            time_hard_start: Hard intraday time starting date

        """
        if type(symbols) is str:
            symbols = [symbols]
        hour = int(str(datetime.datetime.utcnow())[11:13])
        us_business_day = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        if hour < 21:
            last_trading_day = str(pd.to_datetime(date(0)) - us_business_day)[:10]
        else:
            # After trading hour
            last_trading_day = str(pd.to_datetime(date(0)) - us_business_day + us_business_day)[:10]
        last_trading_day = min(last_trading_day, date(0))

        time_start = time.time()
        for i_symbol, symbol in zip(range(len(symbols)), symbols):
            path_pkl = f'{self.dir_price}/{symbol}.pkl'
            pd_listing_entry = self.pd_listing.loc[self.pd_listing.symbol == symbol]
            keys = ['time', 'open', 'high', 'low', 'volume', 'close', 'adjclose', 'symbol']
            is_updated = False
            pd_price_ori = pd.DataFrame({i: [] for i in keys})
            last_date = time_hard_start
            if len(pd_listing_entry) == 0:
                print(f'Symbol - {symbol} - is not listed.')
                is_updated = True
            else:
                ipo_date = max(pd_listing_entry.iloc[0].ipoDate, time_hard_start)
                if (not os.path.isfile(path_pkl)) | force_reload:
                    last_date = unix2date(date2unix(ipo_date)-3600*24)[:10]
                else:
                    pd_price_ori = pd.read_pickle(path_pkl)
                    last_date = pd_price_ori.time.max()[:10]
                    if last_date >= last_trading_day:
                        is_updated = True

                    max_diff_days = pd.to_datetime(pd_price_ori.time).diff().dt.days.max()
                    if max_diff_days > 90:
                        # THere is a problem with the data and it needs to be reloaded
                        last_date = unix2date(date2unix(ipo_date) - 3600 * 24)[:10]
                        is_updated = False
                        pd_price_ori = pd.DataFrame({i: [] for i in keys})


            if not is_updated:
                query_date_start = unix2date(date2unix(last_date) + 3600 * 24)[:10]
                pd_price = self.get_price_range(symbol, date_start=query_date_start, source='online')
                pd_price = pd.concat([pd_price_ori, pd_price]).drop_duplicates()
                pd_price.to_pickle(path_pkl)
            time_span = round(time.time() - time_start, 1)
            print(f'\rTime: {time_span} - {i_symbol + 1}/{len(symbols)}', end='')


self = StockPrice()


if 1==0:
    symbol = 'GLNG'
    base_url = f'https://www.zacks.com/stock/research/{symbol}/earnings-announcements'
    soupdata = make_soup(base_url)
    entry_class_name0 = 'earnings_announcements_earnings_table_wrapper'
    entrys = soupdata.find_all(attrs={'class': entry_class_name0})

    df = pd.read_json('C:/Users/yunso/Desktop/CIK0000006732.json')
    temp = df.to_dict()
    temp1 = temp['facts']['us-gaap']
    temp2 = temp1['AccountsPayable']

