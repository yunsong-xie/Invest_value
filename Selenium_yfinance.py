__author__ = 'Yunsong Xie'
__email__ = 'xiefinance00@gmail.com'
__company__ = 'Xie Finance LLC'


import os, sys, glob
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.service import Service

DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(DIR).replace('\\', '/')+'/Alpha_Vantage')

from lib import *
global service, driver, dict_driver

from selenium.webdriver.chrome.options import Options
chrome_options = Options()
chrome_options.add_argument("--headless")

pd.set_option('display.max_column', 15)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)

pd_login = pd.read_csv('D:/login_info.csv')


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


def initiate_driver(is_login=False):

    def yfinance_login(default_url, is_login):
        # enter login info
        driver.get(default_url)
        if is_login:
            web_wait_element(driver, 'id', 'header-signin-link', time_out=5)
            driver.find_element_by_id('header-signin-link').click()
            username = pd_login.loc[(pd_login.website == 'yahoo finance') & (pd_login.item == 'username')].iloc[0].value
            web_wait_element(driver, 'id', 'login-username', time_out=5)
            driver.find_element_by_id('login-username').send_keys(username)
            driver.find_element_by_id('login-signin').click()

    default_url = 'https://finance.yahoo.com/'
    driver_path= 'static/analysis/chromedriver92.exe'
    service = Service(driver_path)
    service.start()
    driver = webdriver.Remote(service.service_url)
    yfinance_login(default_url)
    dict_driver={'service': service, 'driver': driver}
    return dict_driver


def get_download():
    dir_options = ['C:/Users/yunso/Downloads', 'C:/Users/Yunsong_i7']
    for _ in dir_options:
        if os.path.isdir(_):
            dir = _
            break
    files = [i for i in os.listdir(dir) if i[-4:] == '.csv']
    pd_download = pd.DataFrame({'filename': files})
    if len(pd_download) > 0:
        pd_download['symbol'] = pd_download.filename.str.split('_').str[0]
        pd_download['file_type'] = pd_download.filename.str.split('_').str[-1].str[:-4]
    else:
        pd_download['symbol'] = []
        pd_download['file_type'] = []
    return pd_download


class Recording:
    def __init__(self):
        month = self.time_now()[:7]
        self.csv_record = f'{DIR}/static/selenium/crawling_status/yfinance_{month}.csv'
        if not os.path.isfile(self.csv_record):
            self.pd_record = pd.DataFrame({'symbol': [], 'status': []})
        else:
            self.pd_record = pd.read_csv(self.csv_record)

    def time_now(self):
        return str(datetime.datetime.now())[:19].replace(':', '-')

    def add_symbol(self, symbol, status):
        if symbol not in list(self.pd_record['symbol']):
            pd_new = pd.DataFrame({'symbol': [symbol], 'status': [status]})
            self.pd_record = pd.concat([self.pd_record, pd_new])
            if status == 'Missing':
                self.pd_record.to_csv(self.csv_record, index=False)
        else:
            self.pd_record.loc[self.pd_record.symbol == symbol, 'status'] = status
            self.pd_record.to_csv(self.csv_record, index=False)

    def check_symbol(self, symbol):
        pd_select = self.pd_record.loc[self.pd_record.symbol == symbol]
        if len(pd_select) == 0:
            return 'new'
        else:
            return pd_select.iloc[0]['status']


if __name__ == '__main__0':
    recorder = Recording()
    dict_driver = initiate_driver()
    driver = dict_driver['driver']
    stock_price = StockPrice()
    pd_listing = stock_price.get_listing_info()
    pd_listing = pd_listing.loc[(pd_listing.ipoDate.str[:6] <= '2020-06') & (pd_listing.assetType == 'Stock') &
                                (pd_listing.exchange.isin(['NASDAQ', 'NYSE'])) &
                                ~(pd_listing.symbol.str.contains('-') | pd_listing.symbol.str.contains('\.'))]

    keyword_list = ['Balance', 'Cash']
    symbols = sorted(pd_listing.symbol.unique())
    pd_download = get_download()
    pd_dowload_todo_pre = pd.DataFrame({'symbol': symbols * 3,
                                        'file_type': ['financials'] * len(symbols) +
                                                     ['cash-flow'] * len(symbols) +
                                                     ['balance-sheet'] * len(symbols)})
    pd_dowload_todo_pre = pd_dowload_todo_pre.merge(pd_download, on=['symbol', 'file_type'], how='outer')
    pd_dowload_todo = pd_dowload_todo_pre.loc[pd_dowload_todo_pre.filename.isna()]
    pd_dowload_todo = pd_dowload_todo.sort_values(by=['symbol', 'file_type'], ascending=[True, False])
    symbols = ['AAPL']

    financial_type_list = ['financials', 'balance-sheet', 'cash-flow']
    css_value_list = [f'body {"> div " * 11}> section > div > div',
                      f'body {"> div " * 9}> section > div > div']
    time_start_all = time.time()

    pd_dowload_todo = pd_dowload_todo.loc[pd_dowload_todo.symbol.isin(['CSBR'])]

    for _ind in range(len(pd_dowload_todo)):
        _entry_continue = True
        while _entry_continue:
            try:
                pd_entry = pd_dowload_todo.iloc[_ind]
                symbol, financial_type = pd_entry.symbol, pd_entry.file_type

                if recorder.check_symbol(symbol) != 'Missing':
                    url = f'https://finance.yahoo.com/quote/{symbol}/{financial_type}?p={symbol}'
                    driver.get(url)
                    time.sleep(1)
                    if url != driver.current_url:
                        recorder.add_symbol(symbol, 'Missing')
                    else:
                        recorder.add_symbol(symbol, 'Normal')
                        time_start, _continue, time_span = time.time(), True, 0
                        count = 0
                        while _continue & (time_span < 10):
                            # Select the quarter data
                            item_list = driver.find_elements_by_css_selector(css_value_list[count % 2])
                            if item_list:
                                item_section = [i for i in item_list if 'Income' in i.text][0]
                                span_section = [i for i in item_list if 'Quarterly' in i.text][0]

                                quarter_button = [i for i in span_section.find_elements_by_css_selector('button') if 'Quarterly' in i.text][0]
                                quarter_button.click()
                                _continue = False
                            else:
                                count += 1
                                time.sleep(0.25)
                            time_span = time.time() - time_start


                        time_start_parse, time_span, n_data = time.time(), 0, 0
                        time_threshold = 2
                        while time_span < time_threshold:
                            item_list = driver.find_elements_by_css_selector(css_value_list[count % 2])
                            if item_list:
                                data_section = [i for i in item_list if 'Breakdown' in i.text][0]
                                n_data_parse = len(data_section.find_elements_by_css_selector('div > div > div > div > div > div > span'))
                                if n_data_parse > n_data:
                                    n_data = n_data_parse
                                    time_span = 0
                                    time_start_parse = time.time()
                                else:
                                    time_span = time.time() - time_start_parse
                            else:
                                count += 1
                            time.sleep(0.25)

                        span_section = [i for i in item_list if 'Quarterly' in i.text][0]
                        download_button = [i for i in span_section.find_elements_by_css_selector('button') if 'Download' in i.text][0]
                        download_button.click()
                        time.sleep(2)
                        time_span_all = round(time.time() - time_start_all, 1)
                        print(f'\rtime: {time_span_all} s, {_ind}/{len(pd_dowload_todo)}', end='')
                _entry_continue = False
            except:
                time.sleep(1)


if __name__ == '__main__0':

    path_folder = r'D:\PycharmProjects\Investment\Invest_value\capitalIQ\static\Financial_reports\YF_FR_20210629\raw'
    files = [i.replace('\\', '/') for i in glob.glob(f'{path_folder}/*.csv')]
    pd_file = pd.DataFrame({'path': files})
    pd_file['filename'] = pd_file.path.str.split('/').str[-1].str[:-4]
    pd_file['symbol'] = pd_file['filename'].str.split('_').str[0]
    pd_file['file_section'] = pd_file['filename'].str.split('_').str[-1]
    pd_file['period'] = pd_file['filename'].str.split('_').str[1]
    pd_file = pd_file[['symbol', 'file_section', 'period', 'path']].sort_values(by=['symbol', 'file_section'])
    symbols = sorted(pd_file.symbol.unique())
    time_start = time.time()

    def clean_data(pd_data_list):
        for pd_sec in pd_data_list:
            if len(pd_sec) == 0:
                return pd.DataFrame()
            else:
                keys = [i for i in list(pd_sec.keys()) if i not in ['name', 'ttm', 'section']]
                if len(keys) == 0:
                    return pd.DataFrame()

        pd_data = pd.concat(pd_data_list)
        pd_data['name'] = pd_data['name'].str.replace('\t', '')
        pd_data = pd_data.rename(columns={'name': 'item'})
        keys = list(pd_data.keys())
        keys_front = ['section', 'item']
        keys_back = [i for i in keys if i not in keys_front if i != 'ttm']
        dict_rename = {i: (i[-4:] + '-' + i[:2] + '-' + i[3:5]) for i in keys_back}
        for key in keys_back:
            if str(pd_data[key].dtypes) != 'float64':
                pd_data[key] = pd_data[key].str.replace(',', '').astype(float)
        pd_data = pd_data[keys_front + keys_back].rename(columns=dict_rename)
        return pd_data

    for i_symbol, symbol in zip(range(len(symbols)), symbols):
        pd_select = pd_file.loc[pd_file.symbol == symbol]
        path_output, pd_data_list = f'{os.path.dirname(path_folder)}/{symbol}.csv', []
        if not os.path.isfile(path_output):
            for i in range(len(pd_select)):
                pd_entry = pd_select.iloc[i]
                symbol, file_section, file_path = pd_entry[['symbol', 'file_section', 'path']]

                pd_data_section = pd.read_csv(file_path)
                pd_data_section['section'] = file_section
                pd_data_list.append(pd_data_section)
            time_span = round(time.time() - time_start, 1)
            pd_data = clean_data(pd_data_list)
            print(f'\rTime: {time_span} s - {i_symbol + 1}/{len(symbols)}', end='')
            if len(pd_data) > 0:
                pd_data.to_csv(path_output, index=False)



content = driver.page_source


def find_block(content, keyword, block_shift=None, sep='div'):

    ind_breakdown = content.index(keyword)

    if block_shift:
        n_block_shift, block_level = block_shift
        n_block_shift_abs = abs(n_block_shift)
        if not ((keyword[:4] == f'<{sep}') & (n_block_shift == 0)):
            # In the case where this condition not satisfied, no need to do any pre-processing
            if n_block_shift <= 0:
                content_pre = content[:ind_breakdown]
                anchor_div_pre = re.finditer(f'</*{sep}', content_pre)
                matches, ind_start = [], None
                for _ in anchor_div_pre:
                    matches.append([_.group(), _.start()])
                count_level, count_shift_n = 0, 0
                for i, match in zip(range(len(matches)), matches[::-1]):
                    if (match[0] == f'<{sep}') & (i == 0):
                        # This is still the same block with the keyword
                        count_level = 0
                        if count_level == n_block_shift:
                            count_shift_n = -1
                    else:
                        if match[0] == f'<{sep}':
                            count_level += 1
                        else:
                            count_level += -1
                    if count_level == block_level:
                        count_shift_n += 1
                        if count_shift_n == n_block_shift_abs:
                            ind_start = match[1]
                            break
                if ind_start is None:
                    raise LookupError(f"Can't find the demanded requirments: keyword: {keyword}\n"
                                      f"n_block_shift: {n_block_shift}\n"
                                      f"block_level: {block_level}")
                content_trimmed = content[ind_start:]
            else:
                a=1

    content_trimmed = content[ind_breakdown:]
    anchor_div = re.finditer(f'</*{sep}', content_trimmed)
    count_div, matchs_ind = 0, []
    for _ in anchor_div:
        if _.group() == f'<{sep}':
            count_div += 1
        else:
            count_div += -1
        matchs_ind.append(_.end())
        if count_div == 0:
            ind_end = _.end()
            break
    return content_trimmed[:ind_end]

content_1 = find_block(content, keyword='<div class="Pos(r)"')
