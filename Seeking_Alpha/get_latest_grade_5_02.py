import pyautogui as pygui
import pyperclip, time, os, sys, re, glob, sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pd.set_option("display.max_columns", None)
pd.set_option("display.max_row", 25)
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.width", 5000)

DIR = os.path.dirname(os.path.realpath(__file__))
username = os.getlogin()
DIR_IMAGE = f'{DIR}/Images/{username}'
DIR_LIB = DIR.split('Invest_value')[0] + '/Invest_value/lib'
if DIR_LIB not in sys.path:
    sys.path.append(DIR_LIB)

from image_match import cv_find_all_pic, cv_wait_for_pic

class SAParsing:
    def __init__(self):
        self.dict_url = {
            '497': 'https://seekingalpha.com/screeners/967c3f4d73e9-497'
        }
        self.dict_page_data = {}
        self.loc_default = None
        self.path_db = f'{DIR}/data_output/sa_db.db'
        self.con = sqlite3.connect(self.path_db)

        # pd_temp.to_sql('Test', self.con, if_exists='append', index=False)

    @staticmethod
    def goto_url(url, post_time_sleep=2):
        list_loc = cv_find_all_pic(f'{DIR_IMAGE}/url_box.png')
        pygui.moveTo(list_loc[0][0] + 350, list_loc[0][1] + 12)
        pygui.position()
        pygui.click()
        pygui.hotkey('ctrl', 'a')
        pygui.write(url)
        pygui.press('enter')
        time.sleep(post_time_sleep)

    @staticmethod
    def activate_chrome():
        list_loc = cv_find_all_pic(f'{DIR_IMAGE}/url_box.png')
        pygui.moveTo(list_loc[0][0] + 350, list_loc[0][1] + 12)
        pygui.position()
        pygui.click()

    @staticmethod
    def copy_sa_page_text(time_out=20, scan_period=0.5):
        if 1 == 0:
            time_out, scan_period = 10, 0.5
        n_char_min = 100
        pyperclip.copy('')
        list_loc = cv_find_all_pic(f'{DIR_IMAGE}/Seeking Alpha Logo.png', region=(0, 0, 300, 300))
        self.loc_default = (list_loc[0][0] + 100, list_loc[0][1] + 60)
        pygui.moveTo(self.loc_default[0], self.loc_default[1])
        pygui.click()
        pygui.hotkey('ctrl', 'a')
        time.sleep(0.5)
        text_page, bool_complete = '', False
        time_start, time_span = time.time(), 0
        while (time_span < time_out) & (not bool_complete):
            pygui.hotkey('ctrl', 'c')
            time.sleep(scan_period)
            text_page = pyperclip.paste()
            time_span = time.time() - time_start
            if len(text_page) > n_char_min:
                bool_complete = True

        if len(text_page) < n_char_min:
            raise ValueError(f'Not able to copy the page text')
        return text_page

    def parse_sa_screen_page(self, page_key, post_time_sleep=2):
        if 1 == 0:
            page_key = '497'
            post_time_sleep = 2
        self.reset_chrome_tab()
        self.goto_url(self.dict_url[page_key], post_time_sleep)

        text_page = self.copy_sa_page_text()

        if 'Parse text info':
            text_497_1 = text_page.replace('\r', '')
            _ind_table_start = text_497_1.rindex('\nRank\nSymbol\nCompany Name')
            _ind_table_end = text_497_1.index('Seeking Alpha - Power to Investors\nPower to Investors')
            text_497_table = text_497_1[_ind_table_start: _ind_table_end]

            list_index = [(_.start(0), _.end(0)) for _ in re.finditer(r'\t\n[0-9]+\n', text_497_table)]
            columns = re.findall('[\n\t]*([\w \.\+\-,]+)[\n\t]*', text_497_table[:list_index[0][0]])[1:]

            list_text_chunk = [text_497_table[list_index[_][1]: list_index[_ + 1][0]] for _ in range(len(list_index) - 1)]
            list_text_chunk.append(text_497_table[list_index[-1][1]:])
            list_row_info = []
            for _ind, text_chunk in enumerate(list_text_chunk):
                text_chunk_exe = text_chunk.replace('-\nRating: Not Covered', 'Rating: Not Covered')
                row_info  = re.findall('[\n\t]*([\w \.\+\-,:\(\)&\'\/!\?]+)[\n\t]*', text_chunk_exe)
                assert len(row_info) == len(columns), f'Columns and row info does not match: {row_info}'
                list_row_info.append(row_info)
            pd_info = pd.DataFrame(list_row_info, columns=columns)

        if 'Post clean up':
            list_col_extract = ['Quant Rating', 'SA Analyst Ratings', 'Wall Street Ratings', 'Div Yield']
            for col in list_col_extract:
                _ind_null = pd_info[col].str.contains('-')
                pd_info.loc[_ind_null, col] = '0'
                pd_info[col] = pd_info[col].str.extract('([\d\.]+)').astype(float)
            pd_info['Market Cap Scale'] = 0.001
            pd_info.loc[pd_info['Market Cap'].str[-1] == 'B', 'Market Cap Scale'] = 1
            pd_info.loc[pd_info['Market Cap'].str[-1] == 'T', 'Market Cap Scale'] = 1000
            np_market_cap_scale = pd_info['Market Cap Scale'].values
            pd_info['Market Cap'] = pd_info['Market Cap'].str.extract('([\d\.]+)').astype(float).values.flatten() * np_market_cap_scale
            self.dict_page_data[page_key] = pd_info

        return pd_info

    @staticmethod
    def page_down_confirm(region=None, n_down=3):
        if 1 == 0:
            region, n_down = None, 3
        if region is None:
            width, height = 300, 350
            region = (250, int(pygui.size().height) - 100 - height, width, height)

        list_loc = cv_find_all_pic(f'{DIR_IMAGE}/Seeking Alpha Logo.png', region=(0, 0, 300, 300))
        pygui.moveTo(list_loc[0][0] + 100, list_loc[0][1] + 60)
        pygui.click()

        image_bottom_base = np.asarray(pygui.screenshot(region=region))

        for _ in range(n_down):
            pygui.hotkey('pagedown')
            time.sleep(0.15)

        image_bottom_current = np.asarray(pygui.screenshot(region=region))
        list_loc = cv_find_all_pic(image_bottom_current, find_thresh_hold=0.005, template_filename=image_bottom_base)
        if len(list_loc) == 0:
            message_output = 'Normal Operation'
        else:
            message_output = 'Stop: Page does not scroll'
        if 1 == 0:
            fig, ax = plt.subplots(1, 2, figsize=(12, 9))
            plt.subplot(121), plt.imshow(image_bottom_base, cmap='gray')
            plt.subplot(122), plt.imshow(image_bottom_current, cmap='gray')

        return message_output

    @staticmethod
    def identify_symbol_hist_status(region=None, time_out=20, scan_period=0.75):
        if 1 == 0:
            region, time_out, scan_period = None, 10, 0.75

        if region is None:
            width, height = 400, 250
            region = (250, int(pygui.size().height) - 100 - height, width, height)

        time_start = time.time()
        time_span = 0
        bool_complete = False
        path_image_loading = f'{DIR_IMAGE}/Rating History Loading.png'
        path_image_straight_not_cover = f'{DIR_IMAGE}/Five Straight Not Covered.png'

        while (time_span < time_out) & (not bool_complete):
            time.sleep(scan_period)
            # image_bottom_current = np.asarray(pygui.screenshot(region=region))
            # plt.imshow(image_bottom_current, cmap='gray')
            list_loc_loading = cv_find_all_pic(path_image_loading, region, find_thresh_hold=0.01, bool_output_full_info=True)

            if len(list_loc_loading) == 0:
                bool_complete = True
            else:
                time_span = time.time() - time_start

        if not bool_complete:
            raise ValueError(f'Page is in loading mode for more than {time_out} seconds')

        list_loc_not_covered = cv_find_all_pic(path_image_straight_not_cover, region, find_thresh_hold=0.03)
        if len(list_loc_not_covered) > 0:
            return 'Stop: Five straight not covered.'
        else:
            return 'Normal Operation'

    def get_default_loc(self):
        if self.loc_default is None:
            list_loc = cv_wait_for_pic(f'{DIR_IMAGE}/Seeking Alpha Logo.png', region=(0, 0, 300, 300), find_thresh_hold=0.005, timeout=5)
            self.loc_default = (list_loc[0][0] + 100, list_loc[0][1] + 60)

    def parse_symbol_hist_rating(self, symbol):

        self.reset_chrome_tab()

        url = f'https://seekingalpha.com/symbol/{symbol}/ratings/quant-ratings'
        self.goto_url(url, 2)

        self.get_default_loc()
        bool_hist_strong_buy = self.detect_past_3y_strong_buy()

        if bool_hist_strong_buy:
            count_pagedown, n_max = 0, 20
            bool_continue = True
            while (count_pagedown < n_max) & bool_continue:
                message_page_down = self.page_down_confirm()
                if 'Stop' in message_page_down:
                    bool_continue = False
                else:
                    message_hist_status = self.identify_symbol_hist_status()
                    if 'Stop' in message_hist_status:
                        bool_continue = False

            text_page = self.copy_sa_page_text()
            text_hist = text_page
            if 'Parse text info':
                text_hist_1 = text_hist.replace('\r', '')
                _ind_table_start = text_hist_1.rindex('\nDate\tPrice\tQuant Rating\tQuant Score\t')
                _ind_table_end = text_hist_1.index('\nRatings Summary\n?\nSA Analysts')
                text_hist_table = text_hist_1[_ind_table_start: _ind_table_end]
                text_hist_table = text_hist_table.replace('\n\n', '\n').replace('\n\n', '\n').replace('\n\n', '\n')
                list_index = [(_.start(0), _.end(0)) for _ in re.finditer(r'\n(\d+/\d+/\d+)\t', text_hist_table)]
                columns = re.findall('[\n\t]*([\w \.\+\-,]+)[\n\t]*', text_hist_table[:list_index[0][0]])
                list_text_chunk = [text_hist_table[list_index[_][0]: list_index[_ + 1][0]] for _ in range(len(list_index) - 1)]
                list_text_chunk.append(text_hist_table[list_index[-1][0]:])

                list_row_info = []
                for _ind, text_chunk in enumerate(list_text_chunk):
                    text_chunk_exe = text_chunk.replace('-\nRating: Not Covered', 'Rating: Not Covered')
                    row_info = re.findall('[\n\t]*([\w \.\+\-,:\(\)&\'\/]+)[\n\t]*', text_chunk_exe)

                    if 'Rating: Not Covered' in text_chunk:
                        row_info[2] = 'Not Covered'
                        row_info[3] = '-1'
                        for _ in range(4, len(row_info)):
                            row_info[_] = ''
                        row_info = row_info[:len(columns)]
                    else:
                        assert len(row_info) == len(columns), f'Columns and row info does not match: {row_info}'
                    list_row_info.append(row_info)
                pd_info = pd.DataFrame(list_row_info, columns=columns)

            if 'Post clean up':
                pd_info['Date'] = pd.to_datetime(pd_info['Date'])
                pd_info['Price'] = pd_info['Price'].str.replace(',', '').astype(float)
                pd_info['Quant Score'] = pd_info['Quant Score'].str.extract('([\d\.]+)').astype(float)
                pd_info['Symbol'] = symbol
                cols_head = ['Date', 'Symbol']
                pd_info = pd_info[cols_head + [_ for _ in pd_info.columns if _ not in cols_head]]

        else:
            pd_info = pd.DataFrame()

        return pd_info

    def detect_past_3y_strong_buy(self):

        region = (275, 630, 350, 350)
        # image_bottom_ori = np.asarray(pygui.screenshot(region=region))
        # plt.imshow(image_bottom_ori, cmap='gray')

        list_loc_3y = cv_wait_for_pic(f'{DIR_IMAGE}/Rating History 3Y button.png', region=region, find_thresh_hold=0.005, timeout=3)

        if len(list_loc_3y) == 0:
            list_loc_clicked = cv_wait_for_pic(f'{DIR_IMAGE}/Rating History 3Y button-Clicked.png', region=region, find_thresh_hold=0.005, timeout=3)
            assert len(list_loc_clicked) >= 1
        else:
            list_loc_clicked, n_trials = [], 3
            count_try, bool_success = 0, False
            while (not bool_success) & (count_try < n_trials):
                loc_3y = list_loc_3y[0]
                pygui.moveTo(loc_3y[0] + 13, loc_3y[1] + 60)
                # pygui.moveTo(loc_3y[0] + region[0], loc_3y[1] + region[1])
                pygui.click()
                time.sleep(0.2)
                pygui.moveTo(self.loc_default[0], self.loc_default[1])
                time.sleep(0.4)
                list_loc_clicked = cv_wait_for_pic(f'{DIR_IMAGE}/Rating History 3Y button-Clicked.png', region=region, find_thresh_hold=0.02, timeout=3)
                n_trials += 1
                if len(list_loc_clicked) >= 0:
                    bool_success = True
                else:
                    time.sleep(1)
            assert len(list_loc_clicked) >= 1

        time.sleep(3.5)
        _temp = [list_loc_clicked[0][0] - 143, list_loc_clicked[0][1] + 91]
        region_history_chart = (_temp[0], _temp[1], 1000, 110)

        # region_history_chart = (512, 943, 10, 10)
        image_bottom_ori = np.asarray(pygui.screenshot(region=region_history_chart))
        # plt.imshow(image_bottom_ori, cmap='gray')

        dict_channel_range = {
            0: [26, 43],
            1: [55, 69],
            2: [23, 31],
        }
        image_bottom = image_bottom_ori.copy()

        image_bool = image_bottom[:, :, 0].copy()
        image_bool[:, :] = True
        for ch in dict_channel_range:
            ch_min, ch_max = dict_channel_range[ch]
            image_bool *= (image_bottom[:, :, ch] >= ch_min) * (image_bottom[:, :, ch] <= ch_max)
        bool_strong_buy = image_bool.max() > 0
        return bool_strong_buy

    @staticmethod
    def get_symbols():
        list_symbol_file = glob.glob(f'{DIR}/data_input/Symbols*.csv')
        list_pd_symbol = []
        for symbol_file in list_symbol_file:
            _pd_symbol = pd.read_csv(symbol_file)
            list_pd_symbol.append(_pd_symbol)
        pd_symbol_func = pd.concat(list_pd_symbol)
        pd_symbol_func = pd_symbol_func.drop_duplicates()

        pd_symbol_exist = pd.read_sql('select distinct Symbol from rating', self.con)
        pd_symbol_ignore = pd.read_sql('select distinct Symbol from ignore_rating', self.con)
        list_symbol_exist = list(pd_symbol_exist['Symbol'])
        list_symbol_ignore = list(pd_symbol_ignore['Symbol'])
        pd_symbol_func = pd_symbol_func.loc[~pd_symbol_func['Symbol'].isin(list_symbol_exist + list_symbol_ignore)].copy()
        pd_symbol_func['Quant Rating'] = pd_symbol_func['Quant Rating'].str.extract('([\d\.]+)').astype(float)
        pd_symbol_func = pd_symbol_func.sort_values(by='Quant Rating', ascending=False)

        return pd_symbol_func

    @staticmethod
    def reset_chrome_tab():
        time.sleep(1)
        pygui.hotkey('ctrl', 't')
        time.sleep(0.5)
        pygui.hotkey('ctrl', 't')
        time.sleep(0.5)
        _ = cv_wait_for_pic(f'{DIR_IMAGE}/New Tab.png', region=(0, 0, 800, 35), find_thresh_hold=0.005, timeout=15)
        pygui.hotkey('ctrl', 'l')
        time.sleep(0.5)
        pygui.hotkey('ctrl', 'l')
        time.sleep(0.5)
        _ = cv_wait_for_pic(f'{DIR_IMAGE}/New Tab.png', region=(0, 0, 137, 35), find_thresh_hold=0.005, timeout=15)


self = SAParsing()
if 'C:1' == DIR[:2]:

    pd_symbol = self.get_symbols()
    list_symbol = list(pd_symbol['Symbol'])
    n_symbol_scan = 2000

    list_pd_info_all = []
    symbol = list_symbol[1]
    n_retry_max = 3
    if username == 'ruxie':
        list_symbol_exe = list_symbol[n_symbol_scan:]
    elif username == 'yunso':
        list_symbol_exe = list_symbol[:n_symbol_scan]
    else:
        raise ValueError(f'Not able to identify username: {username}')
    time_start, n_exe = time.time(), len(list_symbol_exe)
    self.activate_chrome()
    for _ind, symbol in enumerate(list_symbol_exe):
        if 1 == 0:
            i_retry, bool_success = 0, False
            while (i_retry < n_retry_max) & (not bool_success):
                try:
                    pd_info_symbol = self.parse_symbol_hist_rating(symbol)
                    bool_success = True
                except Exception as _e:
                    self.reset_chrome_tab()
                    pass
        pd_info_symbol = self.parse_symbol_hist_rating(symbol)
        n_commit = len(pd_info_symbol)
        if len(pd_info_symbol) > 0:
            pd_info_symbol['Symbol'] = symbol
            pd_info_symbol.to_sql('rating', self.con, if_exists='append', index=False)
            list_pd_info_all.append(pd_info_symbol)
        else:
            pd_info_symbol = pd.DataFrame({'Symbol': [symbol]})
            pd_info_symbol.to_sql('ignore_rating', self.con, if_exists='append', index=False)
        time_span = round(time.time() - time_start, 1)
        print(f'Time {time_span} s - progress {_ind + 1}/{n_exe} - {symbol} - entries - {n_commit}')
    pd_info_all = pd.concat(list_pd_info_all)
