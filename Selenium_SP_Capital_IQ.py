
import re, os, sys, traceback, glob
import pandas as pd
import win32gui
from termcolor import colored
from lib import *

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(DIR).replace('\\', '/')+'/Alpha_Vantage')

from lib import Alpha_Vantage
from lib.image_match import *
global DRIVER

from selenium.webdriver.chrome.options import Options
chrome_options = Options()
chrome_options.add_argument("--headless")

pd.set_option('display.max_column', 15)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)

DOWNLOAD_DIR = f'{DIR}\\static\\Financial_reports\\CIQ_FR_{date(0)[:4]}Q{(int(date(0)[5:7]) - 1)//3 + 1}_O'
if not os.path.isdir(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)
DICT_LOGIN = pd.read_csv('D:/login_info.csv').set_index(['website', 'item']).to_dict()['value']
DEFAULT_URL = 'https://libguides.gatech.edu/go.php?c=22546655'

def set_window_front():
    def window_handler(hwnd, top_windows):
        top_windows.append((hwnd, win32gui.GetWindowText(hwnd)))

    for j in range(2):
        results, top_windows = [], []
        win32gui.EnumWindows(window_handler, top_windows)
        for i in top_windows:
            if 'data:, - Google Chrome' in i[1]:
                win32gui.ShowWindow(i[0], 1)
                break


def initiate_driver():

    def capital_iq_login(default_url):

        try:
            # maximize the window
            filename = f'{DIR}/static/pygui/init_blank.png'
            temp = cv_find_pic(filename, region=(0, 0, 1920, 1080), trial=3, wait_time=1)
            pygui.moveTo(500, 500)
            pygui.moveTo(temp[0] + 400, temp[1])
            pygui.doubleClick()

            # enter login info
            DRIVER.get(default_url)
            web_wait_element(DRIVER, 'username', time_out=10)
            login_data = {'username': DICT_LOGIN['gatech', 'username'],
                          'password': DICT_LOGIN['gatech', 'password'],
                          }
            for i in login_data:
                DRIVER.find_element_by_id(i).send_keys(login_data[i])
            DRIVER.find_element_by_name('submit').click()

            filename = f'{DIR}/static/pygui/remember_7_days.png'
            temp = cv_find_pic(filename, region=(0, 0, 1920, 1080), trial=3, wait_time=2)
            pygui.moveTo(500, 500)
            pygui.click(temp[0]-20, temp[1]+6)

            filename = f'{DIR}/static/pygui/send_push.png'
            temp = cv_find_pic(filename, region=(0, 0, 1920, 1080), trial=3, wait_time=1)
            pygui.moveTo(500, 500)
            pygui.click(temp[0], temp[1])
            time.sleep(10)
        except:
            # Somehow
            WebDriverWait(DRIVER, 15).until(EC.presence_of_element_located((By.ID, "ctl03_ciqImage")))

        #Wait for the viewall button to show up
        WebDriverWait(DRIVER, 60).until(EC.presence_of_element_located((By.ID, "ctl03_ciqImage")))
        label = 0
        if label == 1:
            try:
                WebDriverWait(DRIVER, 2).until(EC.presence_of_element_located((By.ID, "readOnlyDialog")))
                read_only_window = DRIVER.find_element_by_id('readOnlyDialog')
                button_OK = read_only_window.find_element_by_class_name('close')
                button_OK.click()
            except Exception as _:
                print('No read only window shown')

            try:
                WebDriverWait(DRIVER, 2).until(EC.presence_of_element_located((By.ID, "onetrust-accept-btn-handler")))
                button_cookies = DRIVER.find_element_by_id('onetrust-accept-btn-handler')
                button_cookies.click()
            except Exception as _:
                print('No accept cookies button')

            try:
                WebDriverWait(DRIVER, 3).until(EC.presence_of_element_located((By.ID, "readOnlyDialog")))
                button_cookies = DRIVER.find_element_by_class_name('close')
                button_cookies.click()
            except Exception as _:
                print('No read-only message')

    driver_path = 'static/analysis/chromedriver92.exe'

    options = webdriver.ChromeOptions()
    options.add_argument("--remote-debugging-port=54123")
    options.add_argument('--disable-infobars')
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-dev-shm-usage")
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_experimental_option('prefs', {'download.default_directory': DOWNLOAD_DIR})
    DRIVER = webdriver.Chrome(driver_path, options=options, service_args=["--verbose"],
                              service_log_path='chromedriver.log')

    set_window_front()
    capital_iq_login(DEFAULT_URL)
    return DRIVER


def web_wait_element(DRIVER, element_id, time_out=5):
    time_delay = 0.15
    time_start = time.time()
    while time.time()-time_start <= time_out:
        try:
            found_item = DRIVER.find_element_by_id(element_id)
            return found_item
        except:
            time.sleep(time_delay)
    return None


def go_to_capital_statement_page(DRIVER, dict_stock_info, init_page=1, count_trial=0):
    time_delay, time_out, n_trials=0.15, 3, 3
    count_trial += 1
    exchange_ticker = dict_stock_info['exchange_ticker']
    if count_trial > n_trials:
        print(colored(f'\nWarning: Sometime went wrong during collect information for {exchange_ticker}'
                      f'. Failed after {n_trials} .', 'yellow'))
        return DRIVER, 'Failed', 'Failed', 'Failed'

    def go_search_symbol(DRIVER):
        time.sleep(0.5)
        found_item = web_wait_element(DRIVER, 'SearchTopBar', time_out=5)
        if found_item is None:
            DRIVER.get(DEFAULT_URL)
            web_wait_element(DRIVER, 'SearchTopBar', time_out=7)
        DRIVER.find_element_by_id('SearchTopBar').send_keys('(' + exchange_ticker + ')')
        time.sleep(0.5)
        DRIVER.find_element_by_id('SearchTopBar').send_keys(Keys.RETURN)

    def identify_current_page(DRIVER, dict_stock_info, time_out=5):
        time_start = time.time()
        symbol = dict_stock_info['symbol']
        while time.time() - time_start <= time_out:
            try:
                element = DRIVER.find_element_by_id('_pageHeader__pageTitle_PageHeaderLabel')
                if exchange_ticker.upper() in element.text.upper():
                    # right company, right page
                    data_info = DRIVER.find_element_by_id('ctl03__tagLine').text
                    if 'except per share items' in data_info:
                        # right company, right page and data is available
                        return 'Balance_sheet'
                    else:
                        # right company, right page, but data is not available, need to reload DRIVER
                        return 'Data_no_available'
                else:
                    # not the right company
                    return 'Random'
            except:
                try:
                    element = DRIVER.find_element_by_id('CompanyHeaderInfo')
                    if exchange_ticker.upper() in element.text.upper():
                        # Right company, not the right page
                        if 'PUBLIC FUND' in element.text.upper():
                            # Its a public fund, no finanical information
                            return 'Public_fund'
                        else:
                            elements = DRIVER.find_elements_by_class_name('ll_link')
                            if 'Balance Sheet' in [_.text for _ in elements]:
                                return 'Company_default'
                            else:
                                return 'Company_default_no_report'
                    else:
                        # not the right company
                        return 'Random'
                except:
                    try:
                        # The company has duplicated entries and go to search page
                        # If so, search the exchange_ticker in the found table and click on it, return Random
                        DRIVER.find_element_by_id('Header_PageHeaderLabel')
                        element = DRIVER.find_element_by_id('ctl01__SearchGridView')
                        element.find_element_by_partial_link_text('(' + exchange_ticker + ')').click()

                        time_label = str(datetime.datetime.now())[:19].replace(':', '-')
                        pygui.screenshot().save(f'{DIR}/static/selenium/crawling_error/{symbol}_{time_label}_DuplicatedEntry.png')

                        return 'Potential_Company_default'
                    except:
                        try:
                            DRIVER.find_element_by_id('_profileHelp_Displaysection1__suggestedResults__searchPrompt')
                            element = DRIVER.find_element_by_class_name('suggestBox')
                            try:
                                element.find_element_by_partial_link_text(exchange_ticker).click()
                                return 'Suggested_page_found_ticker'
                            except:
                                return 'Suggested_page_no_found_ticker'

                        except:
                            try:
                                element = DRIVER.find_element_by_id('__gvctl02__SearchGridView__div')
                                element.find_element_by_partial_link_text(exchange_ticker).click()
                                return 'Suggested_page_found_ticker'
                            except:
                                time_label = str(datetime.datetime.now())[:19].replace(':', '-')
                                pygui.screenshot().save(f'{DIR}/static/selenium/crawling_error/{symbol}_{time_label}_Unknown.png')
                                time.sleep(time_delay)
        # Time out and not the right company, not the right page
        return 'Random'

    if init_page == 1:
        # re-search this symbol
        go_search_symbol(DRIVER)
    elif init_page >= 100:
        # Global initiate the DRIVER
        DRIVER.close()
        DRIVER = initiate_driver()
        print('Problem accurs')
    time.sleep(0.5)
    page_status = identify_current_page(DRIVER, dict_stock_info, time_out)
    company_type = 'company'
    if page_status in ['Balance_sheet', 'Company_default_no_report', 'Data_no_available']:
        company_id = re.findall('COMPANYID=([0-9]*)', DRIVER.current_url.upper())[0]
    elif page_status in ['Suggested_page_no_found_ticker']:
        company_id = '000000000'
    elif page_status == 'Company_default':
        # if this is the company default page, input capital statement page and rerun this function
        company_id = re.findall('COMPANYID=([0-9]*)', DRIVER.current_url.upper())[0]
        capital_statement_url = 'https://www-capitaliq-com.eu1.proxy.openathens.net/CIQDotNet/Financial/BalanceSheet.aspx?CompanyId='+company_id
        DRIVER.get(capital_statement_url)
        time.sleep(1)
        DRIVER, company_id, company_type, page_status = go_to_capital_statement_page(DRIVER, dict_stock_info,
                                                                                     init_page=0, count_trial=count_trial)
    elif page_status=='Public_fund':
        company_id = re.findall('COMPANYID=([0-9]*)', DRIVER.current_url.upper())[0]
        company_type = 'fund'

    elif page_status in ['Potential_Company_default', 'Suggested_page_found_ticker']:
        # redirect from duplicated found entries, similar to Random, but do not need to reinitiate
        time.sleep(1)
        DRIVER, company_id, company_type, page_status = go_to_capital_statement_page(DRIVER, dict_stock_info,
                                                                                     init_page=0, count_trial=count_trial)
    else:
        # if this is the company default page, input capital statement page and rerun this function
        time.sleep(1)
        DRIVER, company_id, company_type, page_status = go_to_capital_statement_page(DRIVER, dict_stock_info,
                                                                                     init_page=1, count_trial=count_trial)

    return DRIVER, company_id, company_type, page_status


def download_section(DRIVER, dict_stock_info, is_view_all=False):
    DRIVER, company_id, company_type, page_status = go_to_capital_statement_page(DRIVER, dict_stock_info, init_page=1,
                                                                                 count_trial=0)

    bold_components = DRIVER.find_elements(By.CSS_SELECTOR, "[class$=' cColBG bold']")
    label_press_release = [i.text for i in bold_components if ('press' in i.text.lower()) & ('release' in i.text.lower())]
    if len(label_press_release) > 0:
        return company_id, company_type, 'press_release'
    tch_class_components = [i for i in DRIVER.find_elements(By.CSS_SELECTOR, "[class='tCH']")[-25:-5] if '-20' in i.text]
    latest_filing_date = max([pd.to_datetime(i.text) for i in tch_class_components])
    current_date = pd.to_datetime(date(0))
    if not ((latest_filing_date.year == current_date.year) & (latest_filing_date.quarter == current_date.quarter)):
        return company_id, company_type, 'quarter_not_matched'

    crawl_status = 'normal'
    if page_status in ['Company_default_no_report']:
        crawl_status = 'No report'
    elif page_status in ['Data_no_available']:
        crawl_status = 'No Data'
    elif page_status == 'Suggested_page_no_found_ticker':
        crawl_status = 'No company found'
    elif company_type == 'company':
        label_button_ok_click = False
        try:
            # Make sure the mode is in "quarterly" query
            filename = f'{DIR}/static/pygui/quarterly.png'
            _ = cv_find_pic(filename, region=(181, 294, 300, 300), trial=3, wait_time=2)
        except:
            filename = f'{DIR}/static/pygui/period_type.png'
            _temp = cv_find_pic(filename, region=(181, 294, 300, 300), trial=3, wait_time=2)
            pygui.moveTo(500, 500)
            pygui.click(_temp[0] + 200, _temp[1] + 10)
            time.sleep(0.25)
            filename = f'{DIR}/static/pygui/quarterly.png'
            _ = cv_find_pic(filename, region=(181, 294, 300, 300), trial=3, wait_time=2)
            pygui.click(_[0] + 50, _[1] + 10)
            label_button_ok_click = True

        try:
            # Make sure to select the original fillings
            filename = f'{DIR}/static/pygui/Latest_fillings.png'
            _ = cv_find_pic(filename, region=(400, 254, 400, 300), trial=2, wait_time=1)
            pygui.click(_[0] + 50, _[1] + 10)
            pygui.moveTo(500, 500)
            time.sleep(0.25)
            filename = f'{DIR}/static/pygui/Original_fillings.png'
            _ = cv_find_pic(filename, region=(400, 254, 400, 300), trial=2, wait_time=1)
            pygui.click(_[0] + 50, _[1] + 10)
            label_button_ok_click = True
        except:
            _a = 1

        try:
            # Make sure to select the original fillings
            filename = f'{DIR}/static/pygui/Standard.png'
            _ = cv_find_pic(filename, region=(200, 250, 400, 300), find_thresh_hold=0.0075, trial=1, wait_time=0.5)
        except:
            filename = f'{DIR}/static/pygui/Template.png'
            _ = cv_find_pic(filename, region=(200, 250, 400, 300), trial=2, wait_time=1)
            pygui.click(_[0] + 150, _[1] + 10)
            pygui.moveTo(500, 500)
            time.sleep(0.25)
            filename = f'{DIR}/static/pygui/Standard.png'
            _ = cv_find_pic(filename, region=(200, 250, 400, 400), trial=2, wait_time=1)
            pygui.click(_[0] + 50, _[1] + 10)
            label_button_ok_click = True

        if label_button_ok_click:
            button_ok = DRIVER.find_element_by_id('_pageHeader_TopGoButton')
            button_ok.click()

        try:
            web_wait_element(DRIVER, element_id='ctl03__rangeSlider_viewAll', time_out=5)
            if is_view_all:
                DRIVER.find_element_by_id('ctl03__rangeSlider_viewAll').click()
                web_wait_element(DRIVER, element_id='ctl03__rangeSlider_viewAll', time_out=5)

            time.sleep(np.random.random() * 1)
            web_wait_element(DRIVER, element_id='_pageHeader__excelReport', time_out=5)
            _ = DRIVER.find_element_by_id('_pageHeader__excelReport').click()
            time.sleep(np.random.random() * 1)
        except:
            data_info = DRIVER.find_element_by_id('ctl03__tagLine').text
            if 'No Data Available' in data_info:
                crawl_status = 'No data'
            else:
                crawl_status = 'Limited period'

    return company_id, company_type, crawl_status


class CiqCrawler:
    def __init__(self):
        date_month = self.time_now[:7]
        self.dict_path = {'ciq_id': f'{DIR}/static/selenium/stock_info/CIQ_company_id.csv',
                          'crawl_status': f'{DIR}/static/selenium/crawling_status/{date_month}.csv', }
        self.pd_ciq_id, self.pd_crawl_info = self.record_crawling_info('init')
        self.dict_ciq_id = {i: list(self.pd_ciq_id[i]) for i in self.pd_ciq_id.keys()}
        self.dict_crawl_status = {i: list(self.pd_crawl_info[i]) for i in self.pd_crawl_info.keys()}

    def record_crawling_info(self, command, dict_crawl_info=None):
        """
        This function records the company ID for CIQ and also record the CIQ crawling status
        These two dataset are stored seperatedly
        Args:
            command (str): Command of the operation
            dict_info (dict): input dict info should contains five keys:
                              exchange_ticker, symbol,company_id, company_type, crawl_status

        """
        if command == 'init':
            if not os.path.isfile(self.dict_path['ciq_id']):
                pd_ciq_id = pd.read_csv(self.dict_path['ciq_id'])
            else:
                pd_ciq_id = pd.DataFrame({'symbol': [], 'exchange_ticker': [], 'company_id': [], 'company_type': []})
            if not os.path.isfile(self.dict_path['crawl_status']):
                pd_crawl_info = pd.DataFrame({'exchange_ticker': [], 'status': []})
            else:
                pd_crawl_info = pd.read_csv(self.dict_path['crawl_status'])
            return pd_ciq_id, pd_crawl_info

        elif command == 'update':
            # Updating exchange ticker company ID info
            if dict_crawl_info['exchange_ticker'] not in self.dict_ciq_id['exchange_ticker']:
                for key in self.dict_ciq_id:
                    self.dict_ciq_id[key].append(dict_crawl_info[key])
                self.pd_ciq_id = pd.DataFrame(self.dict_ciq_id)
                self.pd_ciq_id.to_csv(self.dict_path['ciq_id'], index=False)

            # Updating crawling status information, whether it's successful
            if dict_crawl_info['exchange_ticker'] in self.dict_crawl_status['exchange_ticker']:
                ind = self.dict_crawl_status['exchange_ticker'].index(dict_crawl_info['exchange_ticker'])
                is_force_update = self.dict_crawl_status['status'][ind] != dict_crawl_info['status']
                # Delete the previous result, as there is a confict
                for key in self.dict_crawl_status:
                    self.dict_crawl_status[key] = [self.dict_crawl_status[key][i] for i in range(len(self.dict_crawl_status[key]))
                                                   if i != ind]

            else:
                is_force_update = True
            if is_force_update:
                for key in self.dict_crawl_status:
                    self.dict_crawl_status[key].append(dict_crawl_info[key])
                self.pd_crawl_info = pd.DataFrame(self.dict_crawl_status)
                self.pd_crawl_info.to_csv(self.dict_path['crawl_status'], index=False)
        else:
            raise ValueError(f'Input command {command} is not able to be recognized.')

    def get_missing_xls(self):
        """
        Get the exchange tickers that needs crawling
        Args:
            None
        Returns:
            (list): list of exchange tickers that need crawling
        """
        av_session = Alpha_Vantage.financial_data()
        exchange_ticker_list = list(av_session.pd_ciq.exchange_ticker)

        files = set(os.listdir(DOWNLOAD_DIR))
        exchange_ticker_list_exist = []
        exchange_ticker_set = set(exchange_ticker_list)
        for exchange_ticker in exchange_ticker_set:
            label_found, file_found = False, ''
            info_list = exchange_ticker.replace('.', ':').split(':')
            for file in files:
                file_info = file.split(' ')[1:]
                _bool = all([(i in file_info) for i in info_list])
                if _bool:
                    exchange_ticker_list_exist.append(exchange_ticker)
                    file_found = file
                    label_found = True
                    break
            if label_found:
                files.remove(file_found)
        exchange_ticker_list_crawl_1 = [i for i in exchange_ticker_list if i not in set(exchange_ticker_list_exist)]
        exchange_ticker_list_problem = set(self.pd_crawl_info.loc[self.pd_crawl_info.status != 'normal']['exchange_ticker'])
        exchange_ticker_list_crawl = [i for i in exchange_ticker_list_crawl_1 if i not in exchange_ticker_list_problem]

        pd_yf_earning_new = self.get_crawl_symbols()
        symbols_new = set(pd_yf_earning_new.symbol)

        exchange_ticker_list_crawl_final = [i for i in exchange_ticker_list_crawl if i.split(':')[1] in symbols_new]

        return exchange_ticker_list_crawl_final

    def get_crawl_symbols(self):
        dir_fr = max(glob.glob(f'{DIR}/static/Financial_reports/CIQ_FR_20*'))
        date_first = str(pd.to_datetime(os.path.basename(dir_fr).split('_')[-2]))[:10]

        pd_yf_earning_calendar = StockEarning().get_yf_earning_calendar()
        pd_yf_earning_new = pd_yf_earning_calendar.loc[pd_yf_earning_calendar.date >= date_first]
        return pd_yf_earning_new

    @property
    def time_now(self):
        return str(datetime.datetime.now())[:19].replace(':', '-')


ciq_crawler = CiqCrawler()
self = ciq_crawler

if __name__ == '__main__0':

    DRIVER = initiate_driver()

    exchange_ticker_list = ciq_crawler.get_missing_xls()

    count_crawled, count_reset, time_error_start = 0, 0, 10 ** 9
    time_start = time.time()

    for exchange_ticker in exchange_ticker_list:
        count_crawled += 1
        time_now = str(datetime.datetime.now())[:19]
        symbol = exchange_ticker.split(':')[1]
        dict_stock_info = {'symbol': symbol, 'exchange_ticker': exchange_ticker}
        time_span = str(round(time.time() - time_start, 1))

        print(f'\rCurrent time: {time_now} - Time: {time_span} s. Current progress: {exchange_ticker} - '
              f'{count_crawled} / {len(exchange_ticker_list)}', end='')

        _success, _reset = False, False
        while (not _success) & (count_reset <= 3):
            # Crawling sequence
            try:
                if _reset:
                    if count_reset <= 1:
                        DRIVER.refresh()
                    else:
                        DRIVER = initiate_driver()
                company_id, company_type, crawl_status = download_section(DRIVER, dict_stock_info, is_view_all=True)
                if company_type == 'Fail':
                    _success = False
                else:
                    _success = True
                    time_error_span = time.time() - time_error_start
                    if time_error_span > 1800:
                        # reset time_error_start
                        time_error_start = 10 ** 9
                        count_reset = 0
                    dict_crawl_info = {'exchange_ticker': exchange_ticker, 'symbol': symbol, 'company_id': company_id,
                                       'company_type': company_type, 'status': crawl_status}
                    ciq_crawler.record_crawling_info('update', dict_crawl_info)

            except Exception as _:
                if count_reset == 0:
                    # First error, start counting time_error
                    time_error_start = time.time()
                count_reset += 1
                error_msg = traceback.format_exc()
                print(error_msg)
                print(f'Fatal Error happend for {symbol}')
                _reset = False
                pygui.screenshot().save(f'{DIR}/static/selenium/crawling_error/{symbol}_{ciq_crawler.time_now}_FatalError.png')
                time.sleep(5 + np.random.random() * 5)
                #try:
                #    DRIVER.close()
                #except:
                #    a = 1
            time.sleep(3 + np.random.random() * 6)
        if not _success:
            continue
