# Author: Yunsong Xie
import re, os, time, glob, json
import datetime
from bs4 import BeautifulSoup
import urllib.request
import requests
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from termcolor import colored
from zipfile import ZipFile
import io

from matplotlib import pyplot as plt

from selenium import webdriver

pd.set_option('display.max_column', 15)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)
DIR = os.path.dirname(os.path.abspath(__file__))

def make_soup(url):
    agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'
    headers = {'User-Agent': agent}

    thepage = requests.get(url, headers=headers)
    soupdata = BeautifulSoup(thepage.text, "html.parser")
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


class StockEarning:
    """
    Includes functions to pull earning dates and info
    """
    def __init__(self):
        self.pd_earning_date = None
        self.path_sec = f"{os.path.dirname(DIR)}/static/Financial_reports/SEC"
        self.path_yf_earning = f'{os.path.dirname(DIR)}/static/Financial_reports/Earning_dates.pkl'
        self.path_yf_financial = f'{os.path.dirname(DIR)}/static/Financial_reports/YF_FR'

    def get_yf_earning_dates(self, symbols):
        """
        Get earning table from yahoo finance page for selected symbols.
        Args:
            symbols (str/list): stock symbols to pull the earning dates info

        Returns:
            (pandas.dataframe): The earning dates info of selected symbol(s), columns include:
                symbol, time, eps_estimate, eps_reported, suprise
        """

        path_pickle_earning_dates = self.path_yf_earning
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

    def get_yf_earning_calendar(self, date_start=date(-30), date_end=date(0)):
        """
        Get the yahoo finance earning date from the earning calendar pages, return a pandas dataframe
        Args:
            date_start (str): starting date of the yf_earning calendar, i.e. 2021-05-01
            date_end (str): end date of the yf_earning calendar, i.e. 2021-05-01

        Returns:
            (pandas.dataframe): columns are
            1. symbol: stock symbol
            2. date: yahoo finance earning date
        """

        def parse_symbols(calendar_table_overall, date_current):
            """
            Parse the html for yahoo finance earning date calendar
            Args:
                calendar_table_overall (bs4.element.Tag): the html content in beautiful soup format
                date_current: the current date

            Returns:
                (pandas.dataframe): contains symbol and date
            """
            calendar_table = calendar_table_overall.find(id='cal-res-table')
            calendar_table = calendar_table.find('tbody')
            calendar_entries = calendar_table.find_all('tr')
            symbols_calendar = []
            for calendar_entry in calendar_entries:
                matched = re.findall('title="">(.+)</a>', str(calendar_entry.find('a')))
                if len(matched) > 0:
                    symbols_calendar += matched
            pd_calendar_output = pd.DataFrame({'symbol': symbols_calendar, 'date': [date_current] * len(symbols_calendar)})
            return pd_calendar_output

        date_start_input, date_end_input = date_start, date_end
        path_calendar = f'{os.path.dirname(DIR)}/static/Financial_reports/Calendar/yf_calendar.csv'
        if os.path.isfile(path_calendar):
            pd_calendar_ori = pd.read_csv(path_calendar, sep='\t')
            if date_start <= pd_calendar_ori.date.max():
                date_start = str(pd.to_datetime(pd_calendar_ori.date.max()) + pd.to_timedelta(f'1 day'))[:10]

            if date_end <= pd_calendar_ori.date.min():
                date_end = str(pd.to_datetime(pd_calendar_ori.date.min()) - pd.to_timedelta(f'1 day'))[:10]
        else:
            pd_calendar_ori = pd.DataFrame()

        _date_start, _date_end = pd.to_datetime(date_start), pd.to_datetime(date_end)
        num_day = max((_date_end - _date_start).days, 0)
        calendar_days = [_date_start + pd.to_timedelta(f'{i} day') for i in range(num_day)]
        pd_calendar_list = [pd_calendar_ori]
        time_start = time.time()
        for i_calendar, calendar_day in zip(range(len(calendar_days)), calendar_days):
            search_day_start, search_day_end = str(calendar_day - pd.to_timedelta(f'6 day'))[:10], str(calendar_day)[:10]
            _url = f'https://finance.yahoo.com/calendar/earnings?' \
                   f'from={search_day_start}&to={search_day_end}&day={search_day_end}&size=100'
            soup = make_soup(_url)
            calendar_table_overall = soup.find(id='fin-cal-table')
            entry_num_info = re.findall('of (\d+) results', str(calendar_table_overall))

            if len(entry_num_info) > 0:
                pd_calendar_list.append(parse_symbols(calendar_table_overall, search_day_end))

                entry_num = int(entry_num_info[0])
                n_pages = int(np.ceil(entry_num / 100))

                for i_page in range(n_pages - 1):
                    _url = f'https://finance.yahoo.com/calendar/earnings?' \
                           f'from={search_day_start}&to={search_day_end}&day={search_day_end}&offset={100*(i_page+1)}&size=100'
                    soup = make_soup(_url)
                    calendar_table_overall = soup.find(id='fin-cal-table')
                    pd_calendar_list.append(parse_symbols(calendar_table_overall, search_day_end))
            time_span = round(time.time() - time_start)
            print(f'\rTime: {time_span} s - Complete parsing yahoo-finance Calendar {i_calendar + 1} / {len(calendar_days)}', end='')

        pd_calendar = pd.concat(pd_calendar_list)
        if len(pd_calendar_list) > 1:
            pd_calendar.to_csv(path_calendar, index=False, sep='\t')
        pd_calendar_select = pd_calendar.loc[(pd_calendar.date >= date_start_input) &
                                             (pd_calendar.date <= date_end_input)]
        return pd_calendar_select

    def get_cik_data(self, force_reload=False):
        """
        Obtain cik vs symbol data from sec website
        Args:
            force_reload (Boolean): whether to force to reload

        Returns:
            (pandas.dataframe): corresponding table that includes both cik code and symbol
        """
        if self.pd_cik is None:
            # downloaded from https://www.sec.gov/files/company_tickers.json
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
        #cik = self.get_cik(symbol)
        fzip = ZipFile(zip_file, 'r')
        files = fzip.filelist
        dict_cik_name = {'cik': [], 'name': []}
        time_start = time.time()
        dict_asset_list = []
        for i_file, file in zip(range(len(files)), files):
            f = fzip.open(file, 'r')
            temp = f.read(256).decode('utf-8')
            _info = temp.split('{')[1]
            _info_re = re.findall('cik":(\d+),"entityName":"(.+)","facts', _info)
            if len(_info_re) > 0:
                _cik, _name = _info_re[0]

                f = fzip.open(file, 'r')
                dict_info_1 = json.load(io.StringIO(f.read().decode('utf-8')))
                if 'us-gaap' in dict_info_1['facts']:
                    dict_info = dict_info_1['facts']['us-gaap']
                    if 'Assets' in dict_info:
                        for currency in dict_info['Assets']['units']:
                            dict_asset_currency = [i for i in dict_info['Assets']['units'][currency]]
                            for _ in range(len(dict_asset_currency)):
                                dict_asset_currency[_]['currency'] = currency
                                dict_asset_currency[_]['cik'] = _cik
                            dict_asset_list += dict_asset_currency
                dict_cik_name['cik'].append(_cik)
                dict_cik_name['name'].append(_name)

                time_span = round(time.time() - time_start, 1)
                print(f'\rTime: {time_span} - Progress {symbol}: {i_file + 1}/{len(files)}', end='')
        pd_cik_name = pd.DataFrame(dict_cik_name)
        pd_cik_name.cik = pd_cik_name.cik.str.rjust(10, '0')

        pd_asset = pd.DataFrame(dict_asset_list)
        pd_asset.to_pickle(f'{os.path.dirname(DIR)}/static/Financial_reports/sec_asset.pkl')
        pd_cik_name.to_csv(f'{os.path.dirname(DIR)}/static/Financial_reports/sec_cik_name.csv', index=False)

        fzip.close()

    def _temp1(self):
        pd_asset = pd.read_pickle(f'{os.path.dirname(DIR)}/static/Financial_reports/sec_asset.pkl')
        pd_cik_name = pd.read_csv(f'{os.path.dirname(DIR)}/static/Financial_reports/sec_cik_name.csv')

        pd_asset['cik'] = pd_asset['cik'].astype(int)
        pd_asset['val'] = pd_asset['val'].astype(float)
        pd_asset = pd_asset[['end', 'val', 'currency', 'cik']].drop_duplicates()
        pd_asset = pd_asset.loc[pd_asset.currency == 'USD']
        pd_asset = pd_asset.groupby(['end', 'cik'])['val'].mean().reset_index()
        pd_asset = pd_asset.sort_values(by=['cik', 'end'])
        pd_asset.index = np.arange(len(pd_asset))
        pd_asset = pd_asset.loc[pd_asset.val > 1]
        pd_asset.val = pd_asset.val.abs()

        pd_cik_size = pd_asset.groupby('cik').size().rename('num').reset_index().sort_values(by='cik')
        pd_cik_size['ind_start'] = [0] + list(pd_cik_size.num.cumsum()[:-1])
        pd_cik_size['ind_end'] = list(pd_cik_size.num.cumsum())
        ratio_change_threshold = 3
        pd_asset_end_list = []
        cik_list = list(set(pd_asset.cik))
        time_start = time.time()
        for i_cik, cik in zip(range(len(cik_list)), cik_list):
            ind_start, ind_end = pd_cik_size.iloc[i_cik][['ind_start', 'ind_end']]

            pd_asset_symbol = pd_asset.iloc[ind_start: ind_end] # Takes 21.6 s

            # pd_asset_symbol = pd_asset.loc[pd_asset.cik == cik] # Takes 33.4 s
            if len(pd_asset_symbol) <= 2:
                ind_select_list = range(len(pd_asset_symbol))
                diff_select_list = [0] * len(pd_asset_symbol)
                property_select_list = [f'{len(ind_select_list)}'] * len(ind_select_list)
            else:
                val_array = np.asarray(pd_asset_symbol.val)
                val_diff_array = val_array[:-1] / val_array[1:]
                ind_array = np.arange(len(val_diff_array))
                ind_array_sel = ind_array[((val_diff_array > ratio_change_threshold) | (val_diff_array < (1 / ratio_change_threshold)))]
                if len(ind_array_sel) == 0:
                    ind_select_list = [0, len(val_array) - 1]
                    diff_select_list = val_diff_array[[0, -1]]
                    property_select_list = ['start', 'end']
                else:
                    if 0 in ind_array_sel:
                        ind_array_sel_1 = [i for i in range(len(ind_array_sel)) if i == ind_array_sel[i]]
                    else:
                        ind_array_sel_1 = [0]

                    if (len(val_diff_array) - 1) in ind_array_sel:
                        ind_array_sel_2 = sorted([ind_array_sel[-i - 1] for i in range(len(ind_array_sel))
                                                  if (len(val_diff_array) - 1 - i) == ind_array_sel[-i - 1]])
                    else:
                        ind_array_sel_2 = [len(val_diff_array) - 1]
                    ind_select_list = ind_array_sel_1 + ind_array_sel_2
                    diff_select_list = np.concatenate((val_diff_array[ind_array_sel_1], val_diff_array[[i for i in ind_array_sel_2]]))
                    property_select_list = ['start'] * len(ind_array_sel_1) + ['end'] * len(ind_array_sel_2)


            pd_asset_end_symbol = pd_asset_symbol.iloc[ind_select_list].copy()
            pd_asset_end_symbol['val_diff'] = diff_select_list
            pd_asset_end_symbol['property'] = property_select_list
            pd_asset_end_list.append(pd_asset_end_symbol)
            time_span = round(time.time() - time_start, 1)
            print(f'\rGetting fine price info - time {time_span} s - progress {i_cik + 1}/{len(cik_list)}', end='')
        pd_asset_end = pd.concat(pd_asset_end_list)
        pd_asset_end_outlier = pd_asset_end.loc[(pd_asset_end.val_diff > ratio_change_threshold) |
                                                (pd_asset_end.val_diff < (1 / ratio_change_threshold))]
        np.histogram(pd_asset_end.val_diff, bins=50)

        plt.hist(pd_asset_end.val_diff, bins=50)

    def get_sec_earning_data(self, symbols=None):
        """
        Go to SEC website to pull financial files for selected symbols:
        1. For US companys, only pull 10-Q and 10-K
        2. For foriegn companys, pull all files
        All downloaded files are stored in hard drives
        Args:
            symbols: default: None

        """
        # pd_table_info_all_ori = pd.read_pickle('pd_table_info_all.pkl')
        dir_download = self.path_sec
        if symbols is None:
            symbols = list(pd.read_csv(f'{os.path.dirname(DIR)}/static/Financial_reports/pd_symbol.csv').symbol)
            symbols_error = list(pd.read_csv(f'{os.path.dirname(DIR)}/static/Financial_reports/pd_symbol_error.csv').symbol)
            symbols_completed = [os.path.basename(i)[:-4] for i in glob.glob(f'{dir_download}/*.csv')]
            symbols = [i for i in symbols if i not in set(symbols_completed)]
            symbols = [i for i in symbols if i not in set(symbols_error)]
            symbols_exclude = ['CATM']
            symbols = [i for i in symbols if i not in set(symbols_exclude)]

        default_url = 'https://www.sec.gov/edgar/browse/?CIK={}&owner=exclude'
        driver_path = f'{os.path.dirname(DIR)}/static/analysis/chromedriver92.exe'

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
                    if os.path.isfile(filename_new):
                        os.remove(filename_new)

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
        """

        """
        file_symbols = f'{os.path.dirname(DIR)}/static/Financial_reports/pd_symbol.csv'
        file_symbols_us = f'{os.path.dirname(DIR)}/static/Financial_reports/pd_symbol_us.csv'
        symbols = list(pd.read_csv(file_symbols).symbol)
        symbols_exclude = ['CATM']
        symbols = [i for i in symbols if i not in set(symbols_exclude)]
        symbols_us = []
        n_tol = 31
        time_start, count_us = time.time(), 0
        dict_sum = {'symbol': [], 'match': [], 'yf_total': [], 'sec_total': []}
        dict_detail = {'symbol': [], 'yf_report': [], 'sec_report': [], 'sec_file': []}
        for i_symbol, symbol in zip(range(len(symbols)), symbols):
            # cik = self.get_cik(symbol)
            pd_sec_ori = pd.read_csv(f'{self.path_sec}/{symbol}.csv')
            if '10-Q' in set(pd_sec_ori['Form type']):
                # Tested, companies that has 10-Q never has 20-F filed
                # TODO: Develop program to parse each report from SEC to get the actual first Quarter/Annual filling and report dates
                count_us += 1
                symbols_us.append(symbol)
                pd_sec = pd_sec_ori.loc[pd_sec_ori['Form type'].isin(['10-Q', '10-K'])][['Filing date', 'Reporting date']]
                pd_yf_report = pd.read_csv(f'{self.path_yf_financial}/{symbol}.csv')
                times_yf = [i for i in pd_yf_report.columns if i not in {'section', 'item'}]
                times_sec = pd_sec['Reporting date']
                matched_list = sorted(set(times_sec) & set(times_yf))
                pd_sec_matched = pd_sec.loc[pd_sec['Reporting date'].isin(matched_list)].sort_values(by='Reporting date')
                if len(pd_sec_matched) != len(matched_list):
                    pd_sec_matched = pd_sec_matched.groupby('Reporting date')['Filing date'].min().reset_index()

                dict_detail['symbol'] += [symbol] * len(matched_list)
                dict_detail['yf_report'] += matched_list
                dict_detail['sec_report'] += matched_list
                dict_detail['sec_file'] += list(pd_sec_matched['Filing date'])

                to_rematch_yf_array = np.asarray([i for i in times_yf if i not in set(matched_list)])
                pd_sec_rematch = pd_sec.loc[~pd_sec['Reporting date'].isin(matched_list)]
                to_rematch_sec_file_array = np.asarray(pd_sec_rematch['Filing date'])
                to_rematch_sec_report_array = np.asarray(pd_sec_rematch['Reporting date'])

                count_rematch = 0
                if (len(to_rematch_sec_report_array) > 0) & (len(to_rematch_yf_array) > 0):
                    _n = len(to_rematch_sec_report_array)
                    for ind_report, to_rematch_sec_report in zip(range(_n), to_rematch_sec_report_array):
                        tol_start = unix2date(date2unix(to_rematch_sec_report) - n_tol * 3600 * 24)[:10]
                        tol_end = unix2date(date2unix(to_rematch_sec_report) + n_tol * 3600 * 24)[:10]
                        matched_index = (to_rematch_yf_array >= tol_start) & (to_rematch_yf_array <= tol_end)

                        if any(matched_index):
                            rematched_yf = to_rematch_yf_array[matched_index]

                            if sum(matched_index) > 1:
                                rematched_yf_temp = np.asarray([date2unix(i) for i in rematched_yf])
                                report_date_unix = date2unix(to_rematch_sec_report)
                                close_index = abs(rematched_yf_temp - report_date_unix).argmin()
                                rematched_yf = rematched_yf[close_index][:10]

                            dict_detail['symbol'].append(symbol)
                            dict_detail['yf_report'].append(rematched_yf[0])
                            dict_detail['sec_report'].append(to_rematch_sec_report)
                            dict_detail['sec_file'].append(to_rematch_sec_file_array[ind_report])
                            count_rematch += 1


                n_match = len(matched_list) + count_rematch
                yf_total = len(times_yf)
                sec_total = len(times_sec)
                dict_sum['symbol'].append(symbol)
                dict_sum['match'].append(n_match)
                dict_sum['yf_total'].append(yf_total)
                dict_sum['sec_total'].append(sec_total)

            time_span = round(time.time() - time_start, 1)
            print(f'\rTime: {time_span} - Progress {symbol}: {i_symbol + 1}/{len(symbols)}', end='')

        pd_sum = pd.DataFrame(dict_sum)
        pd_sum['yf_rate'] = pd_sum['match'] / pd_sum['yf_total']
        pd_sum['sec_rate'] = pd_sum['match'] / pd_sum['sec_total']
        pd_sum = pd_sum.sort_values(by='yf_rate')
        pd_detail = pd.DataFrame(dict_detail)

        pd.DataFrame({'symbol': symbols_us}).to_csv(file_symbols_us, index=False)
        pd_detail.to_csv(f'{os.path.dirname(DIR)}/static/Financial_reports/sec_filling_date.csv', index=False)







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
