__author__ = 'Yunsong Xie'
import re, os, sys, datetime
import numpy as np
import pandas as pd
import time
import glob, xlrd
from matplotlib import pyplot as plt
import lib as common_func
import multiprocessing as mp

pd.set_option('display.max_column', 20)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)

DIR = os.path.dirname(os.path.abspath(__file__))
DIR_CACHE = f'{DIR}/cache'
sys.path.append(DIR)
if not os.path.isdir(DIR_CACHE):
    os.makedirs(DIR_CACHE)

dict_time_name = {'Income Statement': 'For the Fiscal Period Ending',
                  'Balance Sheet': 'Balance Sheet as of:',
                  'Cash Flow': 'For the Fiscal Period Ending',
                  'Ratios': 'For the Fiscal Period Ending',
                  'Segments': 'For the Fiscal Period Ending',
                  'Key Stats': 'For the Fiscal Period Ending', }
parse_data = [['Balance Sheet', 'Accounts Receivable'],
              ['Balance Sheet', 'Accumulated Depreciation'],
              ['Balance Sheet', 'Long-Term Debt'],
              ['Balance Sheet', 'Long-Term Leases'],
              ['Balance Sheet', 'Total Current Assets'],
              ['Balance Sheet', 'Total Current Liabilities'],
              ['Balance Sheet', 'Total Equity'],
              ['Balance Sheet', 'Retained Earnings'],
              ['Balance Sheet', 'Total Assets'],
              ['Balance Sheet', 'Total Liabilities'],
              ['Balance Sheet', 'Net Property, Plant & Equipment'],
              ['Balance Sheet', 'Tangible Book Value'],
              ['Balance Sheet', 'Total Cash & ST Investments'],
              ['Balance Sheet', 'Total Shares Out. on Filing Date'],
              ['Cash Flow', 'Income Statement'],
              ['Cash Flow', 'Cash from Ops.'],
              ['Cash Flow', 'Depreciation & Amort., Total'],
              ['Income Statement', 'Total Revenue'],
              ['Income Statement', 'Operating Income'],
              ['Income Statement', 'Revenue'],
              ['Income Statement', 'EBIT'],
              ['Income Statement', 'Gross Profit'],
              ['Income Statement', 'Selling General & Admin Exp.'],
              ['Income Statement', 'Other Operating Expense'],
              ['Income Statement', 'EBITDA'],
              ['Income Statement', 'Diluted EPS'],
              ['Income Statement', 'Net Income']]


type_last_list = ['Total Current Assets', 'Long-Term Leases', 'Long-Term Debt',
                  'Total Current Liabilities', 'Total Equity', 'Total Liabilities', 'Net Property, Plant & Equipment']

item_postive_list = ['Total Cash & ST Investments', 'Total Current Assets', 'Revenue', 'Total Revenue',
                     'Long-Term Debt', 'Long-Term Leases', 'Total Current Liabilities', 'Accounts Receivable',
                     'Net Property, Plant & Equipment']

stock_price = common_func.StockPrice()
SYMBOLS_EXCLUDE = ['RELI', 'PTIX', 'VINO', 'LHC', 'RICE']
#
class ParseExcel:
    def __init__(self):
        self.pd_sum = None

    @staticmethod
    def search_1st_col(sheet, keywords, col_index=0, strict=True):
        if type(keywords) is str:
            keywords=[keywords]

        n_search = 100
        row_keyword = -1

        for keyword in keywords:
            try:
                for i in range(n_search):
                    if strict:
                        if keyword == (sheet.cell(i, col_index).value.strip().split('\n')[0]):
                            row_keyword = i
                            return row_keyword
                    else:
                        if keyword in (sheet.cell(i, col_index).value.strip().split('\n')[0]):
                            row_keyword = i
                            return row_keyword
            except:
                a=1
        return row_keyword

    def get_dict_symbol(self, workbook):
        sheet_HC = workbook.sheet_by_name('Balance Sheet')
        HC_symbol = self.search_1st_col(sheet_HC, 'Financials', strict=False)
        exchange_ticker = re.findall('[A-Za-z]+:[A-Z]+', sheet_HC.cell(HC_symbol, 0).value)[0]
        exchange, symbol = exchange_ticker.split(':')
        dict_symbol = {'symbol': symbol, 'exchange': exchange, 'exchange_ticker': exchange_ticker}
        return dict_symbol

    def get_row_value(self, pd_sum_symbol, workbook, sheet_id_list='Balance Sheet',
                      item_id_list='Total Shares Out. on Filing Date', time_list=None):
        """
        This function output pd dataframe there are two cases:
        1. time_list == []: time_list initiation, query "Share Price" row, get the time_list that has available share Price
        2. time_list != []: Get the row with given name "Net Income", "Revenue", etc. Only select the entries
                            that is within the "time_list" input
        :param pd_sum_symbol:    Current available pd_sum
        :param workbook:    XLRD Workbook
        :param sheet_id:    sheet_id or sheet_name
        :param item_id:     list of row name could belong to this item
        :param time_list:   time_list
        :return:    if time_list is []: pd_sum_output, time_list
                    if time_list is not []: pd_sum_output
        example:
        pd_sum_output, time_list = get_row_value(pd_sum, workbook)
        pd_sum_output, time_list = get_row_value(pd_sum, workbook, sheet_id, item_id='Share Price', time_list=[])
        pd_sum_output = get_row_value(pd_sum, workbook, sheet_id='Income Statement', item_id='Revenue', ['Dec-31-2019'])
        """
        if type(item_id_list) is str:
            item_id_list = [item_id_list]
        if type(sheet_id_list) is str:
            sheet_id_list = [sheet_id_list]

        if not time_list:
            label_base = True
            time_list = []
        else:
            label_base = False

        label_success = 0
        count_parse = -1

        while (label_success == 0) & (count_parse < len(sheet_id_list) - 1):
            count_parse += 1
            sheet_id = sheet_id_list[count_parse]
            item_id = item_id_list[count_parse]

            sheet_HC = workbook.sheet_by_name(sheet_id)
            dict_sum = {'sheet_id': [], 'item_id': [], 'value': [], 'time': []}
            time_row_ind = self.search_1st_col(sheet_HC, dict_time_name[sheet_id])
            value_list_ori = sheet_HC.row_values(time_row_ind)
            pattern = '([A-Za-z]*-[0-9]*-[0-9]*)'
            time_list_temp = [re.search(pattern, i) for i in value_list_ori]

            item_row_num = self.search_1st_col(sheet_HC, item_id)
            if item_row_num != -1:
                # The row does exist
                HC_share_price = sheet_HC.row_values(self.search_1st_col(sheet_HC, item_id))
                if label_base:
                    # Base data query, query "Share Price" for time_list initiation
                    dict_time = {i: time_list_temp[i].group()
                                 for i in range(len(time_list_temp)) if not (time_list_temp[i] is None)}
                    dict_share_price = {i: HC_share_price[i] for i in range(len(HC_share_price))
                                        if not (type(HC_share_price[i]) is str)}
                else:
                    # There is input for time_list query the entries that is within "time_list"
                    dict_time = {i: time_list_temp[i].group()
                                 for i in range(len(time_list_temp)) if not (time_list_temp[i] is None)}
                    dict_time = {i: dict_time[i] for i in dict_time if dict_time[i] in time_list}
                    dict_share_price = {i: HC_share_price[i] for i in dict_time}
            else:
                # The row does not exist
                dict_time = []
                dict_share_price = {}

            time_list = [dict_time[i] for i in set(dict_share_price.keys())]
            item_id_record = item_id_list[0]
            if type(item_id_record) is list:
                item_id_record = item_id_record[0]
            for i in list(dict_share_price.keys()):
                if dict_share_price[i] != '-':
                    dict_sum['sheet_id'].append(sheet_id)
                    dict_sum['item_id'].append(item_id_record.strip())
                    dict_sum['value'].append(dict_share_price[i])
                    dict_sum['time'].append(dict_time[i])

            pd_sum_new = pd.DataFrame(dict_sum)
            if len(pd_sum_new) > 0:
                label_success = 1

        pd_sum_output = pd.concat([pd_sum_symbol, pd_sum_new])

        if label_base:
            return pd_sum_output, time_list
        else:
            return pd_sum_output

    def get_template(self, workbook):
        sheet_HC = workbook.sheet_by_name('Income Statement')
        row_ind=self.search_1st_col(sheet_HC, 'Template:', col_index=1, strict=True)
        template=sheet_HC.cell(row_ind, 2).value
        return template

    def main_parse(self):
        dir_financial_report = max(glob.glob(f'{DIR}/static/Financial_reports/FR_*'))
        file_log = f"{DIR_CACHE}/log_file.txt"
        log_handle = open(file_log, "w")
        log_filter = MyFilter(log_handle)

        count_file = 0
        time_start = time.time()
        date_fr = os.path.basename(dir_financial_report).split('_')[-1]
        xls_files_ori = glob.glob(f'{dir_financial_report}/*.xls')
        count_valid = 0
        pd_sum_list = []
        fail_list = []
        list_symbols = set(stock_price.pd_listing.symbol)
        list_symbols_not_included = []
        miss_income_list = []
        xls_files = xls_files_ori
        xls_files = [i for i in xls_files if 'MEDS' in i]
        for xls_file in xls_files:
            pd_sum_symbol = pd.DataFrame({'sheet_id': [], 'item_id': [], 'value': [], 'time': []})
            count_file += 1
            workbook = xlrd.open_workbook(xls_file, logfile=log_filter)
            sheetname_list = workbook.sheet_names()
            if 'Income Statement' not in sheetname_list:
                miss_income_list.append(xls_file)
            dict_symbol = self.get_dict_symbol(workbook)
            if (dict_symbol['symbol'] not in list_symbols) | (dict_symbol['symbol'] in SYMBOLS_EXCLUDE):
                list_symbols_not_included.append(dict_symbol['exchange_ticker'])
                continue
            ipo_date = stock_price.pd_listing.loc[stock_price.pd_listing.symbol == dict_symbol['symbol']].ipoDate.iloc[0]
            ipo_date = datetime.datetime.strptime(ipo_date, '%Y-%m-%d')
            pd_sum_symbol, time_list = self.get_row_value(pd_sum_symbol, workbook, sheet_id_list='Key Stats',
                                                          item_id_list='Exchange Rate')
            try:
                pd_sum_symbol, time_list = self.get_row_value(pd_sum_symbol, workbook)
                time_list = [i for i in time_list if datetime.datetime.strptime(i,'%b-%d-%Y') >= ipo_date]
                pd_sum_symbol = pd_sum_symbol.loc[pd_sum_symbol.time.isin(time_list)]

                for parse_data_entry in parse_data:
                    sheet_id_list, item_id_list = parse_data_entry
                    pd_sum_symbol = self.get_row_value(pd_sum_symbol, workbook, sheet_id_list=sheet_id_list,
                                                               item_id_list=item_id_list, time_list=time_list)
                count_valid += 1
            except:
                fail_list.append(xls_file)
                continue

            template = self.get_template(workbook)
            pd_sum_symbol['template'] = template

            for key in dict_symbol:
                pd_sum_symbol[key] = dict_symbol[key]

            pd_sum_symbol.time = pd.to_datetime(pd_sum_symbol.time).astype(str).str[:10]


            # Get the Share price information

            symbol = dict_symbol['symbol']
            time_list = sorted(pd_sum_symbol.time.unique())
            if len(time_list) < 4:
                # Only consider files that at least have reported 4 quarters
                continue
            # Get the price info 1 days after the earning date,
            # The date parsed here is the fiscal date, not the actual filling date. Only after round 35 days, the report is available

            # Get the Accurate original filing date
            pd_earning = stock_price.get_earning_dates(symbol)
            pd_earning = pd_earning.loc[pd_earning.time != 'NA']
            earning_time_list = np.sort(pd_earning.time)
            if len(earning_time_list) == 0:
                # There is no earning info returned from yahoo finance
                continue
            dict_time_real = {'time': [], 'time_real': []}
            for earn_time in time_list:
                earning_time_select_list = earning_time_list[earning_time_list > earn_time]
                if len(earning_time_select_list) > 0:
                    dict_time_real['time_real'].append(earning_time_select_list[0])
                    dict_time_real['time'].append(earn_time)
            pd_sum_symbol = pd_sum_symbol.loc[pd_sum_symbol.time.isin(dict_time_real['time'])]
            if len(pd_sum_symbol) == 0:
                # There is no valid financial report dates
                continue

            dict_time_real['time_real'] = [common_func.unix2date(common_func.date2unix(i) + 3600 * 24 * 1)[:10]
                                           for i in dict_time_real['time_real']]
            dict_time_replace = {dict_time_real['time'][i]: dict_time_real['time_real'][i]
                                 for i in range(len(dict_time_real['time_real']))}
            try:
                pd_price = stock_price.get_price_dates(symbol, dates=dict_time_real['time_real'])

            except:
                stock_price.update_price_symbol([symbol], force_reload=True)
                pd_price = stock_price.get_price_dates(symbol, dates=dict_time_real['time_real'])
            if len(pd_price) == 0:
                # In this case it was found that if a stock has stopped trading for 14 days, no result is returned,
                # give up on this stock then
                continue
            pd_price['time_request'] = pd_price['time_request'].replace(dict_time_replace)
            pd_price = pd_price.rename(columns={'close': 'Price'})
            pd_sum_symbol['time'] = pd_sum_symbol['time'].replace(dict_time_replace)

            pd_stock_share_out = pd_sum_symbol.loc[pd_sum_symbol.item_id == 'Total Shares Out. on Filing Date'].copy()
            pd_market_cap_entry = pd_stock_share_out.merge(pd_price[['time', 'Price']], on='time', how='inner')
            pd_market_cap_entry = pd_market_cap_entry.drop_duplicates()
            if str(pd_market_cap_entry.value.dtypes) != 'float64':
                pd_market_cap_entry = pd_market_cap_entry.loc[pd_market_cap_entry.value != 'NA'].copy()
            pd_market_cap_entry['value'] = pd_market_cap_entry['value'] * pd_market_cap_entry['Price']
            pd_market_cap_entry['item_id'] = 'Market Cap'
            pd_market_cap_entry = pd_market_cap_entry[list(pd_stock_share_out.keys())]
            # Done

            pd_sum_symbol_1 = pd.concat([pd_sum_symbol, pd_market_cap_entry]).drop_duplicates()

            workbook.release_resources()
            pd_sum_list.append(pd_sum_symbol_1)
            time_span = round(time.time() - time_start, 1)
            print(f'\rTime: {time_span} - Complete parse file {len(xls_files)} / {count_file} - '
                  f'Valid data entries: {count_valid}', end='')

        log_handle.close()
        pd_sum = pd.concat(pd_sum_list)
        # dict_sum={'exchange': [], 'symbol':[], 'exchange_ticker':[], 'sheet_id':[], 'item':[], 'value':[], 'time':[]}
        pd_sum = pd_sum.drop_duplicates()
        pd_sum = pd_sum.loc[~pd_sum.symbol.isin(SYMBOLS_EXCLUDE)]
        pd_sum['time'] = pd_sum.time.astype(str).str[:10]

        # Add tq (quarter)
        if 'tq' not in list(pd_sum.keys()):
            if '-' not in pd_sum.iloc[0].time[-4:]:
                # Format: 'Jan-01-2020'
                _year = pd_sum.time.str[-4:].astype(int)
                dict_month = {'Jan': 1, 'Feb': 1, 'Mar': 1, 'Apr': 2, 'May': 2, 'Jun': 2,
                              'Jul': 3, 'Aug': 3, 'Sep': 3, 'Oct': 4, 'Nov': 4, 'Dec': 4}
                _quarter = pd_sum.time.str[:3].replace(dict_month)
                pd_sum['tq'] = _year + _quarter / 10
            else:
                # Format: '2020-01-01'
                _year = pd_sum.time.str[:4].astype(int)
                pd_sum['tq'] = ((pd_sum['time'].str[5:7].astype(int) // 4 + 1) / 10) + _year

        # Change market to the currency in the financial report
        pd_sum_mc = pd_sum.loc[pd_sum.item_id == 'Market Cap'].copy()
        pd_sum_exchange = pd_sum.loc[pd_sum.item_id == 'Exchange Rate'].copy().rename(columns={'value': 'exchange_rate'})
        pd_sum_mc['year'] = pd_sum_mc.tq.astype(int)
        pd_sum_exchange['year'] = pd_sum_exchange.tq.astype(int)
        pd_sum_exchange['exchange_rate'] = pd_sum_exchange['exchange_rate'].astype(float)
        pd_sum_exchange = pd_sum_exchange.groupby(['exchange_ticker', 'year'], observed=True)['exchange_rate'].mean().reset_index()
        pd_sum_mc = pd_sum_mc.merge(pd_sum_exchange[['exchange_ticker', 'year', 'exchange_rate']],
                                      on=['exchange_ticker', 'year'], how='inner')
        pd_sum_mc['value'] = pd_sum_mc['value'] / pd_sum_mc['exchange_rate']
        pd_sum_mc = pd_sum_mc[[i for i in list(pd_sum_mc.keys()) if i not in ['year', 'exchange_rate']]]
        pd_sum = pd.concat([pd_sum.loc[pd_sum.item_id!='Market Cap'], pd_sum_mc])

        self.pd_sum = pd_sum

        # Export to pkl
        date_now = common_func.date(0)
        path_parsed_result = f'{DIR}/static/analysis/pd_fr_sum_{date_fr}_{date_now}.pkl'
        pd_sum.to_pickle(path_parsed_result)

        pd_fr_sum_files = [i for i in glob.glob(f'{DIR}/static/analysis/pd_fr_sum_*') if '_final_' not in i]
        data_date_list = set([(os.path.basename(i)[10:18], os.path.basename(i)[19:29]) for i in pd_fr_sum_files])
        pd_temp = pd.DataFrame(data_date_list, columns=['crawl', 'parse'])
        pd_fr_sum_file = pd_temp.groupby('crawl')['parse'].max().reset_index()
        pd_sum_final_list = []
        for i in range(len(pd_fr_sum_file)):
            crawl_date, parse_date = pd_fr_sum_file.iloc[i][['crawl', 'parse']]
            path_parsed_result = f'{DIR}/static/analysis/pd_fr_sum_{crawl_date}_{parse_date}.pkl'
            pd_sum_final_list.append(pd.read_pickle(path_parsed_result))
        pd_sum_final = pd.concat(pd_sum_final_list).drop_duplicates()
        pd_sum_final.to_pickle(f'{DIR}/static/analysis/pd_fr_sum_final_{date_now}.pkl')


        return pd_sum_final

self = ParseExcel()

if __name__ == '__main__1':

    self = ParseExcel()
    pd_sum = self.main_parse()
    symbols = sorted(pd_sum.symbol.unique())
    stock_price.update_price_symbol(symbols)

class MyFilter(object):
    def __init__(self, mylogfile=sys.stdout):
        self.f = mylogfile

    def write(self, data):
        if "WARNING *** OLE2 inconsistency" not in data:
            self.f.write(data)


class StockAnalyze:
    @staticmethod
    def get_pd_sum(path_pd_sum=None):
        if path_pd_sum is None:
            dir_static_analysis = f'{DIR}/static/analysis'
            _files = glob.glob(f'{dir_static_analysis}/pd_fr_sum_final*.pkl')
            path_pd_sum = max(_files)
        pd_sum = pd.read_pickle(path_pd_sum)
        if str(pd_sum.dtypes.value) != 'float64':
            try:
                pd_sum.value = pd_sum.value.astype(float)
            except:
                pd_sum = pd_sum.loc[pd_sum.value != 'NA']
                pd_sum.value = pd_sum.value.astype(float)


        merge_cols = [i for i in list(pd_sum.keys()) if i != 'value']
        pd_sum = pd_sum.groupby(merge_cols, observed=True).value.mean().reset_index()


        return pd_sum

    @staticmethod
    def _convert_year(pd_stock, direction=1):
        """
        Convert the stock info up or down one year, this is for growth rate calculation
        Args:
            pd_stock (pandas.dataframe): stock information dataframe
            direction (int): the direction indicator, if >0, convert up; elif <0, convert down; otherwise, stays

        Returns:
            (pandas.dataframe): the converted dataframe
        """
        year_list = sorted(pd_stock['year'].unique())
        pd_stock_convert = pd_stock.copy()

        if direction < 0:
            pd_stock_convert = pd_stock_convert.loc[~pd_stock_convert['year'].isin(year_list[:abs(direction)])]
            dict_year_replace = {year_list[i + 1]: year_list[i] for i in range(len(year_list) - abs(direction))}
            pd_stock_convert['year'] = pd_stock_convert['year'].replace(dict_year_replace)

        elif direction > 0:
            pd_stock_convert = pd_stock_convert.loc[~pd_stock_convert['year'].isin(year_list[-abs(direction):])]
            dict_year_replace = {year_list[i]: year_list[i + 1] for i in range(len(year_list) - abs(direction))}
            pd_stock_convert['year'] = pd_stock_convert['year'].replace(dict_year_replace)


        return pd_stock_convert

    def get_stock_growth(self, pd_sum_input, item_id_list, n_year=1):
        """
        Calculate growth rate of the specified items. The growth rate is output as the current year comparing to the previous year.
        Args:
            pd_sum_input (pandas.dataframe): The raw stock info dataframe
            item_id_list: The item_ids that need the growth rates calculated
            n_year: Number of years needs the growth rates calculated

        Returns:
            (pandas.dataframe):
        """

        pd_stock = pd_sum_input.loc[pd_sum_input.item_id.isin(item_id_list)].copy()
        if 'year' not in list(pd_stock.keys()):
            pd_stock['year'] = (pd_stock['tq'] // 1).astype(int)

        pd_stock = pd_stock.groupby(['exchange_ticker', 'item_id', 'year'], observed=True)['value'].mean().reset_index()
        ind_abs = pd_stock.item_id.isin(item_postive_list)
        pd_stock.loc[ind_abs, 'value'] = pd_stock.loc[ind_abs, 'value'].abs()
        pd_stock_pre = self._convert_year(pd_stock, direction=-1)

        pd_stock_merge = pd_stock_pre.merge(pd_stock, on=['exchange_ticker', 'year', 'item_id'], how='inner')
        pd_stock_merge['growth'] = 0

        pd_stock_merge_p_ = pd_stock_merge.loc[(pd_stock_merge['value_x'] > 0)]
        pd_stock_merge_n_ = pd_stock_merge.loc[(pd_stock_merge['value_x'] < 0)]
        pd_stock_merge_z_ = pd_stock_merge.loc[(pd_stock_merge['value_x'] == 0)]
        pd_stock_merge__z = pd_stock_merge.loc[(pd_stock_merge['value_x'] == 0)]

        pd_stock_merge_pp = pd_stock_merge_p_.loc[pd_stock_merge_p_['value_y'] > 0].copy()
        pd_stock_merge_pn = pd_stock_merge_p_.loc[pd_stock_merge_p_['value_y'] < 0].copy()
        pd_stock_merge_np = pd_stock_merge_n_.loc[pd_stock_merge_n_['value_y'] > 0].copy()
        pd_stock_merge_nn = pd_stock_merge_n_.loc[pd_stock_merge_n_['value_y'] < 0].copy()
        pd_stock_merge_zp = pd_stock_merge_z_.loc[pd_stock_merge_z_['value_y'] > 0].copy()
        pd_stock_merge_zn = pd_stock_merge_z_.loc[pd_stock_merge_z_['value_y'] < 0].copy()
        pd_stock_merge_pz = pd_stock_merge__z.loc[pd_stock_merge__z['value_x'] > 0].copy()
        pd_stock_merge_nz = pd_stock_merge__z.loc[pd_stock_merge__z['value_x'] < 0].copy()


        pd_stock_merge_pp['growth'] = pd_stock_merge_pp['value_x'] / pd_stock_merge_pp['value_y'] - 1
        pd_stock_merge_pn['growth'] = -1
        pd_stock_merge_np['growth'] = 2
        pd_stock_merge_nn['growth'] = ((pd_stock_merge_nn['value_x'] / pd_stock_merge_nn['value_y'] - 1) *
                                       np.sign(pd_stock_merge_nn['value_x'] - pd_stock_merge_nn['value_y']))
        pd_stock_merge_zp['growth'] = 1
        pd_stock_merge_zn['growth'] = -1
        pd_stock_merge_pz['growth'] = -1
        pd_stock_merge_nz['growth'] = 1

        pd_stock_merge = pd.concat([pd_stock_merge_pp, pd_stock_merge_pn, pd_stock_merge_np, pd_stock_merge_nn,
                                    pd_stock_merge_zp, pd_stock_merge_zn, pd_stock_merge_pz, pd_stock_merge_nz])
        year_list = sorted(pd_stock['year'].unique())
        dict_year_replace = {year_list[i]: year_list[i + 1] for i in range(len(year_list) - 1)}
        pd_stock_merge['year'] = pd_stock_merge['year'].replace(dict_year_replace)

        pd_stock_final = pd_stock.merge(pd_stock_merge[['exchange_ticker', 'item_id', 'year', 'growth']],
                                        on=['exchange_ticker', 'item_id', 'year'], how='inner')
        if n_year > 1:
            merge_cols = ['exchange_ticker', 'item_id', 'year']
            pd_stock_final_pre_copy = pd_stock_final.copy()
            for i_year in range(n_year - 1):
                pd_stock_final_pre = self._convert_year(pd_stock_final_pre_copy, direction=1)
                pd_stock_final_pre_copy = pd_stock_final_pre.copy()
                growth_key = f'growth-{i_year + 1}'
                pd_stock_final_pre = pd_stock_final_pre.rename(columns={'growth': growth_key})[merge_cols+[growth_key]]
                pd_stock_final = pd_stock_final.merge(pd_stock_final_pre, on=merge_cols, how='inner')

        return pd_stock_final

    @staticmethod
    def get_stock_ratio(pd_sum, item_id_x, item_id_y, col_val='value'):

        merge_cols = ['exchange_ticker', 'tq', 'time']
        pd_sum_select = pd_sum.loc[pd_sum.item_id.isin([item_id_x, item_id_y])]
        pd_x = pd_sum_select.loc[pd_sum_select.item_id == item_id_x][merge_cols + [col_val]].rename(columns={col_val: 'temp1'})
        pd_y = pd_sum_select.loc[pd_sum_select.item_id == item_id_y][merge_cols + [col_val]].rename(columns={col_val: 'temp2'})
        pd_xy = pd_x.merge(pd_y, on=merge_cols, how='inner')
        pd_xy[f'{item_id_y} / {item_id_x}'] = pd_xy['temp2'] / pd_xy['temp1']
        col_name = f'{item_id_y} / {item_id_x}'
        pd_xy = pd_xy[merge_cols + [col_name]]

        pd_sum_output = pd_sum[merge_cols].drop_duplicates()
        pd_sum_output = pd_sum_output.merge(pd_xy, on=merge_cols, how='inner')

        return pd_sum_output, col_name

    @staticmethod
    def get_year_data(pd_data, val_cols, data_type='latest'):
        pd_data['year'] = (pd_data['tq'] // 1).astype(int)
        if data_type == 'latest':
            pd_filter = pd_data.groupby(['exchange_ticker', 'year'])['tq'].max().reset_index()
            pd_output = pd_data.merge(pd_filter, on=['exchange_ticker', 'tq', 'year'], how='inner')
            pd_output = pd_output.groupby(['exchange_ticker', 'year'])[val_cols].mean().reset_index()
        else:
            pd_output = pd_data.groupby(['exchange_ticker', 'year'])[val_cols].mean().reset_index()
        return pd_output

    def get_cap_growth(self, pd_sum, pd_data_1, n_year=2):
        pd_cap_growth = self.get_stock_growth(pd_sum, item_id_list=['Market Cap'], n_year=n_year)
        pd_cap_growth['year'] = pd_cap_growth['year'] - n_year
        cap_growth_col_list = []
        for i in range(n_year):
            if i == 0:
                pd_cap_growth = pd_cap_growth.rename(columns={'growth': 'cap_growth'})
                cap_growth_col_list.append('cap_growth')
            else:
                pd_cap_growth = pd_cap_growth.rename(columns={f'growth-{i}': f'cap_growth-{i}'})
                cap_growth_col_list.append(f'cap_growth-{i}')
        merge_cols = ['exchange_ticker', 'year']
        pd_cap_growth = pd_data_1.merge(pd_cap_growth[merge_cols+cap_growth_col_list], on=merge_cols, how='inner')
        return pd_cap_growth


def get_pd_growth(pd_sum_select, dict_extra_info, dict_item_id, n_year=2, mp_id=0, mp_queue=None):

    item_id_list = list(dict_item_id.values())
    stock_an = StockAnalyze()
    time_start = time.time()
    pd_item_growth_list = []
    exchange_ticker_list = list(pd_sum_select.exchange_ticker.unique())
    exchange_ticker_list = exchange_ticker_list
    for i_ticker, exchange_ticker in zip(range(len(exchange_ticker_list)), exchange_ticker_list):
        pd_stock = pd_sum_select.loc[pd_sum_select.exchange_ticker == exchange_ticker]
        time_list = sorted(pd_stock.time.unique(), key=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
        pd_item_growth_sec_list = []
        for i_time in range(len(time_list) - (n_year + 1) * 4 + 1):
            time_sec_list = time_list[i_time: (i_time + (n_year + 1) * 4)]
            dict_year = {i: time_sec_list[-1] for i in time_sec_list[-4:]}
            for _ in range(n_year):
                dict_year_pre = {i: time_sec_list[_*4+3] for i in time_sec_list[(_*4):(_*4+4)]}
                dict_year.update(dict_year_pre)
            pd_stock_sec = pd_stock.loc[pd_stock.time.isin(time_sec_list)].copy()
            pd_stock_sec['year'] = pd_stock_sec['time']
            pd_stock_sec['year'] = pd_stock_sec['year'].replace(dict_year)

            pd_stock_sec_sec = stock_an.get_stock_growth(pd_stock_sec, item_id_list=item_id_list, n_year=n_year)
            pd_item_growth_sec_list.append(pd_stock_sec_sec)
        pd_item_growth_entry = pd.concat(pd_item_growth_sec_list)

        pd_item_growth_list.append(pd_item_growth_entry)
        if mp_queue is not None:
            mp_queue.put({mp_id: (i_ticker + 1) / len(exchange_ticker_list)})
        else:
            time_span = round(time.time() - time_start, 1)
            print(f'\rTime: {time_span} - Progress: {len(exchange_ticker_list)}/{i_ticker + 1}', end='')

    pd_item_growth = pd.concat(pd_item_growth_list)
    for key in dict_extra_info:
        pd_item_growth = pd_item_growth.merge(dict_extra_info[key], on=['exchange_ticker', 'year'], how='inner')

    dict_pd_data = {key: pd_item_growth.loc[pd_item_growth.item_id == dict_item_id[key]] for key in dict_item_id}

    merge_cols = ['exchange_ticker', 'year']
    keys_temp = sorted(dict_pd_data.keys())
    pd_data = pd.DataFrame()
    for i_key, key in zip(range(len(keys_temp)), keys_temp):
        _pd_data = dict_pd_data[key]
        growth_keys = [_ for _ in list(_pd_data.keys()) if 'growth' in _]
        dict_rename = {_: f'{_}_{key}' for _ in growth_keys}
        _pd_data = _pd_data.rename(columns=dict_rename)
        _pd_data = _pd_data[merge_cols + list(dict_rename.values())]
        if i_key == 0:
            pd_data = _pd_data
        else:
            pd_data = pd_data.merge(_pd_data, on=merge_cols, how='inner')
    pd_data = pd_data.merge(pd_item_growth[merge_cols+['PB']].drop_duplicates(), on=merge_cols, how='inner')
    pd_data_growth = pd_data.copy()
    if mp_queue is not None:
        mp_queue.put(pd_data_growth)

    return pd_data_growth


def mp_print_progress(mp_queue, time_start, dict_progress):
    if not mp_queue.empty():
        _result = mp_queue.get()
        if type(_result) is dict:
            dict_progress['progress'].update(_result)
        else:
            dict_progress['result'].append(_result)
    time_span = round(time.time() - time_start, 1)
    progress = round(min(dict_progress['progress'].values()) * 100)
    print(f'\rTime: {time_span} s - Progress: {progress}%', end='')
    return dict_progress


if __name__ == '__main__0':

    stock_an = StockAnalyze()
    try:
        temp = pd_sum.iloc[0]
    except:
        pd_sum = stock_an.get_pd_sum()

    n_quarters_min = 12

    dict_item_id = {'A': 'Total Current Assets', 'B': 'Cash from Ops.',
                    'C': 'Tangible Book Value',
                    'D': 'Total Cash & ST Investments', 'E': 'Gross Profit'}
    pd_item_id = pd.DataFrame(dict_item_id, index=['col']).T.reset_index().rename(columns={'index': 'code'})
    item_id_list = list(dict_item_id.values())
    item_id_include_list = item_id_list + ['Tangible Book Value', 'Market Cap', 'Total Shares Out. on Filing Date']
    pd_sum = pd_sum.loc[(pd_sum.item_id.isin(item_id_include_list)) & (pd_sum.tq > 2003)]

    pd_exchange_ticker = pd_sum[['exchange_ticker', 'time']].drop_duplicates().groupby('exchange_ticker').size()
    pd_exchange_ticker = pd_exchange_ticker.loc[pd_exchange_ticker > n_quarters_min].rename('num').reset_index()
    pd_exchange_ticker = pd_exchange_ticker.sort_values(by='num')
    exchange_ticker_list = list(pd_exchange_ticker.exchange_ticker)
    pd_sum_select = pd_sum.loc[pd_sum.exchange_ticker.isin(exchange_ticker_list)]
    # np.random.shuffle(exchange_ticker_list)
    extra_info = {'Tangible Book Value': 'PB', 'Total Current Assets': 'MC_TCA'}
    dict_extra_info = {}
    for key in extra_info:

        pd_pb, ratio_col_name = stock_an.get_stock_ratio(pd_sum_select, item_id_x=key,
                                                         item_id_y='Market Cap', col_val='value')
        pd_pb = pd_pb.rename(columns={ratio_col_name: extra_info[key], 'time': 'year'})
        pd_pb = pd_pb[['exchange_ticker', 'year', extra_info[key]]]
        dict_extra_info[extra_info[key]] = pd_pb

    mp_queue = mp.Queue()
    n_process = 8
    n_year = 2
    time_start = time.time()
    dict_progress = {'progress': {i: 0 for i in range(n_process)}, 'result': [], 'completed': [n_process, 0]}
    for i in range(n_process):
        n_total, n_sec = len(exchange_ticker_list), len(exchange_ticker_list) // n_process + 1
        exchange_ticker_list_sec = exchange_ticker_list[i::n_process]
        pd_sum_sec = pd_sum_select.loc[pd_sum_select.exchange_ticker.isin(exchange_ticker_list_sec)]
        mp.Process(target=get_pd_growth, args=(pd_sum_sec, dict_extra_info, dict_item_id, n_year, i, mp_queue)).start()

    while len(dict_progress['result']) != n_process:
        time.sleep(0.25)
        dict_progress = mp_print_progress(mp_queue, time_start, dict_progress)

    pd_data_growth = pd.concat(dict_progress['result']).rename(columns={'year':'time'})

    # add share and book value info
    keys = ['Total Shares Out. on Filing Date', 'Tangible Book Value']
    pd_sum_mc = pd_sum.loc[pd_sum.item_id.isin(keys)]
    for key_ori in keys:
        if key_ori == 'Total Shares Out. on Filing Date':
            key = 'Shares'
        elif key_ori == 'Tangible Book Value':
            key = 'Book Value'
        else:
            key = key_ori
        if key not in list(pd_data_growth.keys()):
            pd_sum_mc_entry = pd_sum_mc.loc[pd_sum_mc.item_id == key_ori]
            dict_rename = {'value': key}
            pd_sum_mc_entry = pd_sum_mc_entry[['exchange_ticker', 'time', 'value']].rename(columns=dict_rename)
            pd_data_growth = pd_data_growth.merge(pd_sum_mc_entry, on=['exchange_ticker', 'time'], how='inner')
    pd_data_growth = pd_data_growth.merge(dict_extra_info['MC_TCA'].rename(columns={'year': 'time'}),
                                          on=['exchange_ticker', 'time'], how='inner')

    pd_data_growth.to_pickle(f'{DIR}/static/analysis/pd_data_grow_{common_func.date(0)}.pkl')
    pd_item_id.to_pickle(f'{DIR}/static/analysis/pd_data_grow_item_id_{common_func.date(0)}.pkl')


def bu_se_2_profit(pd_stock_bu_se):
    pd_stock_act = pd_stock_bu_se.loc[~((pd_stock_bu_se.time > pd.to_datetime(common_func.date(-hold_day_min))) &
                                        (pd_stock_bu_se.status == -1))][['time', 'symbol', 'status']].copy()
    #pd_stock_act = pd_stock_bu_se


    # Add price information to the buying stocks
    pd_stock_act['time'] = pd_stock_act['time'].astype(str).str[:10]
    pd_stock_act_price_list = []
    symbols = sorted(pd_stock_act.symbol.unique())
    time_start = time.time()
    dates = list(pd_stock_act.time.unique())
    for i_symbol, symbol in zip(range(len(symbols)), symbols):
        pd_symbol = pd_stock_act.loc[pd_stock_act.symbol == symbol]
        pd_price = stock_price.get_price_dates(symbol, dates)
        pd_price['time'] = pd_price['time_request']
        pd_price = pd_price[['time', 'close']].rename(columns={'close': 'price'})
        pd_symbol = pd_symbol.merge(pd_price, on='time', how='outer')
        pd_symbol['symbol'] = symbol
        pd_symbol = pd_symbol.fillna(0)
        pd_stock_act_price_list.append(pd_symbol)
        time_span = round(time.time() - time_start, 1)
        print(f'\rTime: {time_span} - Add price info - Progress: {len(symbols)}/{i_symbol + 1}', end='')
    print()

    pd_stock_act = pd.concat(pd_stock_act_price_list)
    pd_stock_act = pd_stock_act.sort_values(by=['time', 'symbol'])
    pd_stock_act.index = np.arange(len(pd_stock_act))
    pd_stock_act = pd_stock_act.loc[pd_stock_act.price > 0]

    dates = sorted(pd_stock_act.time.unique())
    money = 100
    time_start = time.time()
    dict_stock = {'symbol': [], 'n': []}
    dict_reward = {'money': [], 'date': [], 'n_stock': []}
    for i_date, date in zip(range(len(dates)), dates):
        pd_act = pd_stock_act.loc[pd_stock_act.time == date]
        pd_act_bu = pd_act.loc[pd_act.status == 1]
        pd_act_se = pd_act.loc[pd_act.status == -1]
        set_stock_bu = set(pd_act_bu.symbol)
        set_stock_se = set(pd_act_se.symbol)
        if len(dict_stock['symbol']) == 0:
            if len(pd_act_bu) == 0:
                break
            money_each = money / len(pd_act_bu)
            dict_stock = {'symbol': [], 'n': []}
            for i in range(len(pd_act_bu)):
                symbol, price = pd_act_bu.iloc[i][['symbol', 'price']]
                dict_stock['symbol'].append(symbol)
                dict_stock['n'].append(money_each / price)
        else:
            pd_stock = pd.DataFrame(dict_stock)
            pd_stock = pd_stock.merge(pd_act[['symbol', 'price']].drop_duplicates(), on='symbol', how='inner')
            money = (pd_stock['price'] * pd_stock['n']).sum()
            stock_list = dict_stock['symbol']
            dict_stock = {'symbol': [], 'n': []}
            if len(set_stock_se) != 0:
                stock_list = [i for i in stock_list if i not in set_stock_se]
            if len(set_stock_bu) != 0:
                stock_list = stock_list + list(set_stock_bu)
            if len(stock_list) > 0:
                money_each = money / len(stock_list)
                for symbol in stock_list:
                    pd_act_entry = pd_act.loc[pd_act.symbol == symbol].iloc[0]
                    price = pd_act_entry['price']
                    dict_stock['symbol'].append(symbol)
                    dict_stock['n'].append(money_each / price)

        dict_reward['money'].append(money)
        dict_reward['date'].append(date)
        dict_reward['n_stock'].append(len(dict_stock['symbol']))

        time_span = round(time.time() - time_start, 1)
        print(f'\rTime: {time_span} - Trading execution - Progress: {len(dates)}/{i_date + 1}', end='')
    print()
    pd_stock_final = pd.DataFrame(dict_stock)
    pd_reward = pd.DataFrame(dict_reward)
    pd_reward['date'] = pd.to_datetime(pd_reward['date'])
    fig, _ax = plt.subplots(1, 1)
    ax = fig.axes
    ax[0].plot(pd_reward.date, pd_reward.money, '.')
    ax[0].set_yscale('log')
    print((money/100)**(1/14))
    return pd_stock_final, pd_reward


def bu_se_2_profit_fine(pd_stock_bu_se_fine, n_stock_max=5):

    time_act_list = list(pd_stock_bu_se_fine.loc[pd_stock_bu_se_fine.act].time)
    pd_stock_bu_se_fine_temp = pd_stock_bu_se_fine.loc[pd_stock_bu_se_fine.time.isin(time_act_list)]
    pd_stock_act_sim = pd_stock_bu_se_fine_temp[['time', 'symbol', 'Price', 'weight_final']]
    pd_stock_act_sim = pd_stock_act_sim.set_index(['time', 'symbol']).T
    dict_stock_act_ori = pd_stock_act_sim.to_dict()
    dict_stock_act = {}
    dates, symbols = sorted(pd_stock_bu_se_fine['time'].unique()), list(pd_stock_bu_se_fine['symbol'].unique())
    for date in dates:
        for symbol in symbols:
            if (date, symbol) in dict_stock_act_ori:
                if date in dict_stock_act:
                    dict_stock_act[date].update({symbol: dict_stock_act_ori[(date, symbol)]})
                else:
                    dict_stock_act[date] = {symbol: dict_stock_act_ori[(date, symbol)]}

    time_start = time.time()
    dict_holding = {'cash': 100}
    asset = 100
    dict_reward = {'asset': [], 'date': [], 'n_stock': [], 'hold_num': [], 'hold_val': []}
    dates = sorted(dict_stock_act.keys())
    #dates = ['2009-11-06', '2009-11-09']
    for i_date, date in zip(range(len(dates)), dates):
        dict_act_ori = dict_stock_act[date]
        stock_value = sum([dict_holding[i] * dict_act_ori[i]['Price'] for i in dict_holding
                           if (i != 'cash') & (dict_holding[i] > 0)])
        asset = stock_value + dict_holding['cash']
        dict_act_keys = sorted(list(dict_act_ori.keys()), key=lambda x: -dict_act_ori[x]['weight_final'])[:n_stock_max]
        dict_act = {i: dict_act_ori[i] for i in dict_act_keys}

        weight_final_total = sum([dict_act[i]['weight_final'] for i in dict_act])
        if weight_final_total > 0:
            weight_unit = asset / weight_final_total
            for symbol in dict_holding:
                dict_holding[symbol] = 0
            for symbol in dict_act:
                price = dict_act[symbol]['Price']
                weight_final = dict_act[symbol]['weight_final']
                dict_holding[symbol] = weight_unit * weight_final / price
            dict_holding['cash'] = 0
        else:
            for symbol in dict_holding:
                dict_holding[symbol] = 0
            dict_holding['cash'] = asset
        dict_holding_num = {i:dict_holding[i] for i in dict_holding if (i != 'cash') & (dict_holding[i] > 0)}
        dict_holding_val = {i: (dict_holding[i] * dict_act[i]['Price']) for i in dict_holding
                            if (i != 'cash') & (dict_holding[i] > 0)}
        dict_reward['asset'].append(asset)
        dict_reward['date'].append(date)
        dict_reward['n_stock'].append(len(dict_act))
        dict_reward['hold_num'].append(dict_holding_num)
        dict_reward['hold_val'].append(dict_holding_val)

        time_span = round(time.time() - time_start, 1)
        print(f'\rTime: {time_span} - Trading execution - Progress: {len(dates)}/{i_date + 1}', end='')
    print()

    pd_reward = pd.DataFrame(dict_reward)
    pd_reward['date'] = pd.to_datetime(pd_reward['date'])
    return pd_reward


def get_trading_weight(growth, pb, min_growth=0.3, max_growth=10, growth_slope=20, max_pb_1=25, max_pb_2=45):
    weight_cal = ((1 + (np.minimum(np.maximum(growth, min_growth), max_growth) - min_growth)
                   / (max_growth - min_growth) * growth_slope) *
                  ((max_pb_1 - np.maximum(np.minimum(pb, max_pb_2), max_pb_1)) / (max_pb_2 - max_pb_1) + 1))
    return weight_cal


if __name__ == '__main__':

    n_stock_max = 6

    try:
        a = pd_data_growth.iloc[0]
    except:
        path_growth = max(glob.glob(f'{os.path.dirname(DIR)}/static/analysis/pd_data_grow_2*.pkl'))
        pd_data_growth = pd.read_pickle(path_growth)
        pd_data_growth = pd_data_growth.rename(columns={'year': 'time'})

    dict_keys = {'Total Shares Out. on Filing Date': 'Shares', 'Tangible Book Value': 'Book Value'}
    if len([i for i in list(dict_keys.values()) if i not in list(pd_data_growth.keys())]) > 0:
        try:
            temp = pd_sum.iloc[0]
        except:
            stock_an = StockAnalyze()
            pd_sum = stock_an.get_pd_sum()

        pd_sum_mc = pd_sum.loc[pd_sum.item_id.isin(dict_keys)]
        for key in dict_keys:
            if key not in list(pd_data_growth.keys()):
                pd_sum_mc_entry = pd_sum_mc.loc[pd_sum_mc.item_id == key]
                dict_rename = {'value': dict_keys[key]}
                pd_sum_mc_entry = pd_sum_mc_entry[['exchange_ticker', 'time', 'value']].rename(columns=dict_rename)
                pd_data_growth = pd_data_growth.merge(pd_sum_mc_entry, on=['exchange_ticker', 'time'], how='inner')

    pd_data_growth['symbol'] = pd_data_growth['exchange_ticker'].str.split(':').str[-1]

    symbols = sorted(pd_data_growth.symbol.unique())
    pd_data_plot_0 = pd_data_growth.loc[pd_data_growth.symbol.isin(symbols)].copy()

    pd_data_plot_1 = pd_data_plot_0.loc[(pd_data_plot_0['growth_C'] > 0.5) & (pd_data_plot_0['growth-1_C'] > 0.2)].copy()
    #pd_data_plot_1 = pd_data_plot_1.loc[(pd_data_plot_1['growth_C'] > pd_data_plot_1['growth-1_C'])]
    pd_data_plot_1 = pd_data_plot_1.loc[(pd_data_plot_1['growth_A'] > pd_data_plot_1['growth-1_A'])]

    #pd_data_plot_1 = pd_data_plot_1.loc[(pd_data_plot_1['growth_A'] > 0.3) & (pd_data_plot_1['growth-1_A'] > 0.3)].copy()
    #pd_data_plot_1 = pd_data_plot_1.loc[(pd_data_plot_1['PB'] < 25) & (pd_data_plot_1['PB'] > 5.5)]
    pd_data_plot_1 = pd_data_plot_1.loc[(pd_data_plot_1['MC_TCA'] < 20) & (pd_data_plot_1['MC_TCA'] > 5.5)]
    #pd_data_plot_1 = pd_data_plot_1.loc[(pd_data_plot_1['PB'] < 25)]

    pd_data_plot_1 = pd_data_plot_1.loc[(pd_data_plot_1['growth_A'] >= 0.0) & (pd_data_plot_1['growth-1_A'] >= 0.0)]
    pd_data_plot_1 = pd_data_plot_1.loc[(pd_data_plot_1['growth_B'] >= 0.0) & (pd_data_plot_1['growth-1_B'] >= 0.0)]
    pd_data_plot_1 = pd_data_plot_1.loc[(pd_data_plot_1['growth_C'] >= 0.0) & (pd_data_plot_1['growth-1_C'] >= 0.0)]
    pd_data_plot_1 = pd_data_plot_1.loc[(pd_data_plot_1['growth_D'] >= 0.0) & (pd_data_plot_1['growth-1_D'] >= 0.0)]
    pd_data_plot_1 = pd_data_plot_1.loc[(pd_data_plot_1['growth_E'] >= 0.0) & (pd_data_plot_1['growth-1_E'] >= 0.0)]

    pd_data_plot_1 = pd_data_plot_1.sort_values(by=['time', 'exchange_ticker'])

    # pd_stock_hold = pd_data_plot_1[['time', 'exchange_ticker', 'symbol']].copy()
    pd_stock_hold = pd_data_plot_1.copy()
    pd_stock_hold['status'] = 1

    hold_day_min = 370

    pd_stock_act_list = []
    symbols = sorted(pd_stock_hold.symbol.unique())

    pd_stock_hold = pd_stock_hold.loc[pd_stock_hold.symbol.isin(symbols)]
    pd_stock_hold['time'] = pd.to_datetime(pd_stock_hold['time'])
    dates = sorted(pd_stock_hold.time.unique())

    pd_price_fine_list = []
    pd_stock_act_init_list = []
    time_start = time.time()
    # Get the dates that sells stocks becaues
    for i_symbol, symbol in zip(range(len(symbols)), symbols):

        pd_symbol = pd_stock_hold.loc[pd_stock_hold.symbol == symbol]

        # Get the buy/sell actions based on financial reports
        if len(pd_symbol) > 1:
            pd_time_diff = pd_symbol.time.diff().reset_index()
            pd_time_diff['time'] = list(pd_time_diff['time'])[1:] + [np.nan]
            index_selling = pd_time_diff.loc[pd_time_diff.time.dt.days >= hold_day_min]['index']
            pd_selling = pd_symbol.loc[index_selling].copy()

            pd_selling['status'] = -1

            # Compile buying options
            pd_buying = pd_symbol.copy()
            bins = np.asarray(
                [pd_symbol.time.min()] + list(pd_selling.time + pd.to_timedelta('1 days')) + [pd_symbol.time.max()]).astype(
                np.datetime64)
            bin_n, bin_loc = np.histogram(pd.to_datetime(pd_symbol.time), bins=bins)
            anchor_list = []
            for i in range(len(bin_n)):
                anchor_list += [bin_loc[i]] * bin_n[i]
            pd_buying['sec'] = anchor_list
            pd_buying['status'] = pd_buying['sec'].rank(method='dense')
            pd_buying['status'] = (pd_buying['status'].shift(axis=0) != pd_buying['status']).astype(int)
            pd_buying = pd_buying[[i for i in list(pd_buying.keys()) if i != 'sec']]
            pd_buying = pd_buying.loc[pd_buying.status != 0]
        else:
            pd_selling = pd_symbol.copy()
            pd_buying = pd_symbol.copy()
            pd_selling['status'] = -1
            pd_buying['status'] = 1
        pd_selling['time'] = (pd_selling.time + pd.to_timedelta(f'{hold_day_min} days'))
        #if ((pd_symbol.iloc[-1].time < pd.to_datetime(common_func.date(-hold_day_min))) & (len(pd_selling) < len(pd_buying))):
        if len(pd_selling) < len(pd_buying):
            pd_selling_final = pd_symbol.iloc[[-1]].copy()
            pd_selling_final['time'] = pd_symbol.iloc[-1].time + pd.to_timedelta(f'{hold_day_min} days')
            pd_selling_final['status'] = -1
        else:
            pd_selling_final = pd.DataFrame()
        pd_stock_act_init = pd.concat([pd_selling, pd_buying, pd_selling_final]).sort_values(by='time')
        pd_stock_act_init_list.append(pd_stock_act_init)


        # Make all time in pd_stock_act_init a trading date
        date_start = str(pd_stock_act_init.time.min())[:10]
        if pd_stock_act_init.iloc[-1].status == -1:
            date_end = str(pd_stock_act_init.time.max() + pd.to_timedelta('14 days'))[:10]
        else:
            date_end = common_func.date(0)
        pd_price = stock_price.get_price_range(symbol, date_start, date_end)[['time', 'close']]
        pd_price['time'] = pd_price['time'].str[:10]
        time_price_list = np.asarray(pd_price.time)
        time_list = []
        for i in pd_stock_act_init.time.astype(str).str[:10]:
            if i in time_price_list:
                time_list.append(i[:10])
            elif i < time_price_list[-1]:
                time_list.append(time_price_list[time_price_list > i][0][:10])
            else:
                time_list.append(time_price_list[-1][:10])
        pd_stock_act_init['time'] = pd.to_datetime(time_list)
        pd_price = pd_price.loc[(pd_price.time >= min(time_list)) & (pd_price.time <= max(time_list))]

        # Fine tune trading actions based on price
        date_start = str(pd_stock_act_init.time.min())[:10]
        if pd_stock_act_init.iloc[-1].status == -1:
            date_end = str(pd_stock_act_init.time.max() + pd.to_timedelta('14 days'))[:10]
            bins_act = np.sort(pd_stock_act_init.time)
            if len(pd_selling_final) == 0:
                bins_info = np.sort(pd_stock_act_init.time)
            else:
                pd_stock_act_init_last = pd_stock_act_init.time.iloc[:1].copy()
                pd_stock_act_init_last.loc[:] = pd.to_datetime(date_end)
                bins_info = np.sort(pd.concat([pd_symbol.time, pd_stock_act_init_last]))
        else:
            date_end = common_func.date(0)
            pd_stock_act_init_last = pd_stock_act_init.time.iloc[:1].copy()
            pd_stock_act_init_last.loc[:] = pd.to_datetime(date_end)
            bins_act = np.sort(pd.concat([pd_stock_act_init.time, pd_stock_act_init_last]))
            bins_info = np.sort(pd.concat([pd_symbol.time, pd_stock_act_init_last]))

        pd_price['time'] = pd.to_datetime(pd_price['time'])

        bin_n, bin_loc = np.histogram(pd_price['time'], bins=bins_act)
        sign_list, _sign = [], 1
        for i in range(len(bin_n)):
            sign_list += [_sign] * bin_n[i]
            _sign = -_sign
        pd_price['sec'] = sign_list
        _time_list = list(pd_stock_act_init.loc[pd_stock_act_init.status == -1].time)
        pd_price.loc[pd_price.time.isin(_time_list), 'sec'] = -1

        bin_n, bin_loc = np.histogram(pd_price['time'], bins=bins_info)
        time_info_list = []
        for i in range(len(bin_n)):
            time_info_list += [bins_info[i]] * bin_n[i]
        pd_price['time_financial'] = time_info_list

        pd_price = pd_price.rename(columns={'close': 'Price'})
        dict_rename = {'time': 'time_financial'}
        pd_price = pd_price.merge(pd_symbol[['time', 'growth_C', 'growth-1_C', 'Book Value', 'Shares']].rename(columns=dict_rename),
                                  on='time_financial', how='inner')
        pd_price['PB'] = pd_price['Shares'] * pd_price['Price'] / pd_price['Book Value']


        pd_price_sec_bu = pd_price.loc[pd_price.sec == 1]
        if len(pd_stock_act_init.loc[pd_stock_act_init.status == -1]) > 0:
            _time_list = list(pd_stock_act_init.loc[pd_stock_act_init.status == -1].time)
            pd_price_fine_se = pd_price.loc[pd_price.time.isin(_time_list)]

            pd_price_fine_symbol = pd.concat([pd_price_sec_bu, pd_price_fine_se]).copy().sort_values(by='time')
        else:
            pd_price_fine_symbol = pd_price


        pd_price_fine_symbol['weight'] = get_trading_weight(pd_price_fine_symbol['growth_C'], pd_price_fine_symbol['PB'])
        sec_weight_array = pd_price_fine_symbol[['sec', 'weight']].values
        weight_pre = -1
        weight_list, act_list, weight = [], [], 0
        for i in range(len(pd_price_fine_symbol)):
            is_act = False
            sec, weight = sec_weight_array[i]
            if weight > 0:
                # Previously hold some stock
                if sec == -1:
                    # Have to sell
                    weight = 0
                    is_act = True
                else:
                    if weight_pre == -1:
                        is_act = True
            else:
                if sec == -1:
                    is_act = True
            if is_act:
                weight_pre = weight
            weight_list.append(weight_pre)
            act_list.append(is_act)


        pd_price_fine_symbol['weight_final'] = weight_list
        pd_price_fine_symbol['symbol'] = symbol
        pd_price_fine_symbol['act'] = act_list

        pd_price_fine_list.append(pd_price_fine_symbol)
        time_span = round(time.time() - time_start, 1)
        print(f'\rTime: {time_span} - Obtaining fine trading strategy {len(symbols)} / {i_symbol + 1}', end='')
    print()

    pd_stock_bu_se_fine_ori = pd.concat(pd_price_fine_list).copy()
    pd_stock_bu_se_fine_ori['time'] = pd_stock_bu_se_fine_ori['time'].astype(str).str[:10]
    pd_stock_bu_se_fine_ori.loc[pd_stock_bu_se_fine_ori.weight_final == -1, 'weight_final'] = 0
    pd_stock_bu_se_fine_ori['time_financial'] = pd_stock_bu_se_fine_ori['time_financial'].astype(str).str[:10]
    pd_stock_bu_se_fine_ori.loc[pd_stock_bu_se_fine_ori.PB <= 2, 'weight_final'] = 0
    #pd_weight = pd_stock_bu_se_fine.groupby('time').weight_final.sum()
    #pd_weight = pd_weight.loc[pd_weight>0]
    #pd_stock_bu_se_fine = pd_stock_bu_se_fine.loc[pd_stock_bu_se_fine.time.isin(list(pd_weight.index))].copy()
    pd_stock_bu_se_fine_ori = pd_stock_bu_se_fine_ori.sort_values(by=['time', 'symbol'])
    time_act_list = list(pd_stock_bu_se_fine_ori.loc[pd_stock_bu_se_fine_ori.act].time)
    pd_stock_bu_se_fine = pd_stock_bu_se_fine_ori.loc[pd_stock_bu_se_fine_ori.time.isin(time_act_list)]

    pd_reward = bu_se_2_profit_fine(pd_stock_bu_se_fine=pd_stock_bu_se_fine, n_stock_max=n_stock_max)

    fig, _ax = plt.subplots(1, 1)
    ax = fig.axes
    ax[0].plot(pd_reward.date, pd_reward.asset, '.')
    ax[0].set_yscale('log')
    n_year = (pd_reward.date.iloc[-1]-pd_reward.date.iloc[0]).days/365
    gain = (pd_reward.asset.iloc[-1]/pd_reward.asset.iloc[0])
    print( gain ** (1 / n_year))

    pd_stock_act_init_final = pd.concat(pd_stock_act_init_list).sort_values(by=['time', 'symbol'])

    symbols = list(pd_stock_bu_se_fine.symbol.unique())
    pd_temp_list = []
    for symbol in symbols:
        pd_temp = pd_stock_bu_se_fine.loc[pd_stock_bu_se_fine.symbol == symbol]
        pd_temp_diff = pd_temp.weight_final.diff()
        pd_temp_list.append(pd_temp.loc[pd_temp_diff.loc[(pd_temp_diff.abs()>0)|(pd_temp_diff.isna())].index])
    pd_stock_temp_final = pd.concat(pd_temp_list).sort_values(by=['time', 'symbol'])

    usd_total = 90000
    _dict_recommend_val, _dict_recommend_num = pd_reward.iloc[-2].hold_val, pd_reward.iloc[-2].hold_num
    _keys = list(_dict_recommend_val.keys())
    dict_recommend = {'symbol': _keys, 'usd': [_dict_recommend_val[i] for i in _dict_recommend_val],
                      'num': [_dict_recommend_num[i] for i in _dict_recommend_num]}
    pd_recommend = pd.DataFrame(dict_recommend)
    _rate = usd_total / pd_recommend.usd.sum()
    pd_recommend[['usd', 'num']] = pd_recommend[['usd', 'num']] * _rate
    pd_recommend = pd_recommend.sort_values(by='usd', ascending=False)
    pd_recommend['usd_int'] = 0
    pd_recommend['num_int'] = pd_recommend['num'].astype(int)
    pd_recommend['usd_int'] = pd_recommend['num_int'] / pd_recommend['num'] * pd_recommend['usd']
    pd_recommend = pd_recommend.set_index('symbol')
    pd_recommend_int = pd_recommend[['num_int', 'usd_int']]
    pd_recommend_int['usd_int'].sum()


if 1==0:
    pd_temp_16 = pd_stock_bu_se_fine_ori[['symbol', 'time', 'weight_final']].rename(columns={'weight_final': 'weight_final_16'})
    pd_temp_24 = pd_stock_bs_ori[['symbol', 'time', 'weight_final']].rename(columns={'weight_final': 'weight_final_24'})
    pd_merge = pd_temp_16.merge(pd_temp_24, on=['symbol', 'time'], how='outer')
    pd_merge.loc[pd_merge.weight_final_24 != pd_merge.weight_final_16]