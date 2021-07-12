__author__ = 'Yunsong Xie'
import re, os, sys, datetime
import numpy as np
import pandas as pd
import time
import glob, xlrd
from matplotlib import pyplot as plt
import lib as common_func
import multiprocessing as mp

pd.set_option('display.max_column', 15)
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
              ['Balance Sheet', 'Currency'],
              ['Balance Sheet', 'Long-Term Debt'],
              ['Balance Sheet', 'Long-Term Leases'],
              ['Balance Sheet', 'Total Current Assets'],
              ['Balance Sheet', 'Total Current Liabilities'],
              ['Balance Sheet', 'Total Equity'],
              ['Balance Sheet', 'Retained Earnings'],
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
              ['Income Statement', 'Total Assets'],
              ['Income Statement', 'Revenue'],
              ['Income Statement', 'EBIT'],
              ['Income Statement', 'Gross Profit'],
              ['Income Statement', 'Selling General & Admin Exp.'],
              ['Income Statement', 'Other Operating Expense'],
              ['Income Statement', 'EBITDA'],
              ['Income Statement', 'Diluted EPS'],
              ['Income Statement', 'Net Income']]


parse_data = [['Balance Sheet', 'Long-Term Debt'],
               ['Key Stats', 'Exchange Rate']]

type_last_list = ['Total Current Assets', 'Long-Term Leases', 'Long-Term Debt',
                  'Total Current Liabilities', 'Total Equity', 'Total Liabilities', 'Net Property, Plant & Equipment']

item_postive_list = ['Total Cash & ST Investments', 'Total Current Assets', 'Revenue', 'Total Revenue',
                     'Long-Term Debt', 'Long-Term Leases', 'Total Current Liabilities', 'Accounts Receivable',
                     'Net Property, Plant & Equipment']

stock_price = common_func.StockPrice()
SYMBOLS_EXCLUDE = ['RELI', 'PTIX', 'VINO', 'LHC', 'RICE']
#
class ParseExcel:
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
        dir_financial_report = f'{DIR}/Financial_reports'
        file_log = f"{DIR_CACHE}/log_file.txt"
        log_handle = open(file_log, "w")
        log_filter = MyFilter(log_handle)

        count_file = 0
        time_start = time.time()
        xls_files = glob.glob(f'{dir_financial_report}/*.xls')
        count_valid = 0
        pd_sum_list = []
        fail_list = []
        list_symbols = set(stock_price.pd_listing.symbol)
        list_symbols_not_included = []
        miss_income_list = []
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
            pd_stock_share_out = pd_sum_symbol.loc[pd_sum_symbol.item_id == 'Total Shares Out. on Filing Date'].copy()
            symbol = dict_symbol['symbol']
            time_list = sorted(pd_sum_symbol.time.unique())
            # Get the price info one day after the earning date
            time_list_price = [common_func.unix2date(common_func.date2unix(i) + 3600 * 24)[:10] for i in time_list]
            dict_time_replace = {time_list_price[i]: time_list[i] for i in range(len(time_list))}
            try:
                pd_price = stock_price.get_price_dates(symbol, dates=time_list_price)
                pd_price['time'] = pd_price['time_request'].replace(dict_time_replace)
            except:
                stock_price.update_price_symbol([symbol], force_reload=True)
                pd_price = stock_price.get_price_dates(symbol, dates=time_list_price)
                pd_price['time'] = pd_price['time_request'].replace(dict_time_replace)
            pd_price = pd_price.rename(columns={'close': 'Price'})
            pd_market_cap_entry = pd_stock_share_out.merge(pd_price[['time', 'Price']], on='time', how='inner')
            pd_market_cap_entry['value'] = pd_market_cap_entry['value'] * pd_market_cap_entry['Price']
            pd_market_cap_entry['item_id'] = 'Market Cap'
            pd_market_cap_entry = pd_market_cap_entry[list(pd_stock_share_out.keys())]
            # Done

            pd_sum_symbol = pd.concat([pd_sum_symbol, pd_market_cap_entry])

            workbook.release_resources()
            pd_sum_list.append(pd_sum_symbol)
            time_span = round(time.time() - time_start, 1)
            print('\rTime: ' + str(time_span) + ' - Complete parse file ' + str(count_file) + '/' + str(len(xls_files)) +
                  ' - Valid data entries: ' + str(count_valid), end='')

        log_handle.close()
        os.remove(file_log)
        pd_sum = pd.concat(pd_sum_list)
        # dict_sum={'exchange': [], 'symbol':[], 'exchange_ticker':[], 'sheet_id':[], 'item':[], 'value':[], 'time':[]}
        pd_sum = pd_sum.drop_duplicates()
        pd_sum = pd_sum.loc[~pd_sum.symbol.isin(SYMBOLS_EXCLUDE)]

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
        pd_sum_exchange = pd_sum_exchange.groupby(['exchange_ticker', 'year'], observed=True)['exchange_rate'].mean().reset_index()
        pd_sum_mc = pd_sum_mc.merge(pd_sum_exchange[['exchange_ticker', 'year', 'exchange_rate']],
                                      on=['exchange_ticker', 'year'], how='inner')
        pd_sum_mc['value'] = pd_sum_mc['value'] / pd_sum_mc['exchange_rate']
        pd_sum_mc = pd_sum_mc[[i for i in list(pd_sum_mc.keys()) if i not in ['year', 'exchange_rate']]]
        pd_sum = pd.concat([pd_sum.loc[pd_sum.item_id!='Market Cap'], pd_sum_mc])

        # Export to pkl
        date_now = common_func.date(0)
        path_parsed_result = f'{DIR}/static/pd_sum_{date_now}.pkl'
        pd_sum.to_pickle(path_parsed_result)



        return pd_sum


if __name__ == '__main__1':
    self = ParseExcel()
    pd_sum = self.main_parse()


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
            dir_static = f'{DIR}/static'
            _files = glob.glob(f'{dir_static}/pd_sum*.pkl')
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


if __name__ == '__main__1':
    stockan = StockAnalyze()
    pd_sum = stockan.get_pd_sum()
    pd_stock_sum = stockan.get_stock_growth(pd_sum, item_id_list=['Total Current Assets', 'Total Revenue', 'Cash from Ops.',
                                                                  'Gross Profit',
                                                                  'Total Cash & ST Investments'], n_year=3)

    pd_mc_asset, ratio_col_name = stockan.get_stock_ratio(pd_sum, 'Total Current Assets', 'Market Cap', col_val='value')
    pd_mc_asset = stockan.get_year_data(pd_mc_asset, val_cols=ratio_col_name, data_type='latest')

    dict_item_id = {'A': 'Total Current Assets', 'B': 'Cash from Ops.',
                    'C': 'Total Cash & ST Investments', 'D': 'Gross Profit'}
    dict_pd_data = {key: pd_stock_sum.loc[pd_stock_sum.item_id == dict_item_id[key]] for key in dict_item_id}

    merge_cols = ['exchange_ticker', 'year']
    keys_temp = sorted(dict_pd_data.keys())
    pd_data = pd.DataFrame()
    for i_key, key in zip(range(len(keys_temp)), keys_temp):
        _pd_data = dict_pd_data[key]
        growth_keys = [_ for _ in list(_pd_data.keys()) if 'growth' in _]
        dict_rename = {_:f'{_}_{key}' for _ in growth_keys}
        _pd_data = _pd_data.rename(columns=dict_rename)
        _pd_data = _pd_data[merge_cols + list(dict_rename.values())]
        if i_key == 0:
            pd_data = _pd_data
        else:
            pd_data = pd_data.merge(_pd_data, on=merge_cols, how='inner')


    pd_data_plot_1 = pd_data.loc[(pd_data['growth-1_A'] > 0.1) & (pd_data['growth-2_A'] > 0.1)]
    pd_data_plot_1 = stockan.get_cap_growth(pd_sum, pd_data_plot_1, n_year=1)
    pd_data_plot_1 = pd_data_plot_1.merge(pd_mc_asset, on=['exchange_ticker', 'year'], how='inner')
    pd_data_plot_1 = pd_data_plot_1.loc[pd_data_plot_1['year'] > 2005]
    #pd_data_plot_1 = common_func.clean_outliers(pd_data_plot_1, ['growth_A'], sigma=5)
    pd_data_plot_1 = common_func.clean_outliers(pd_data_plot_1, ['cap_growth'], sigma=5)
    pd_data_plot_1 = common_func.clean_outliers(pd_data_plot_1, ['Market Cap / Total Current Assets'], sigma=5)
    pd_data_plot_0 = pd_data_plot_1.copy()
    pd_data_plot_1 = pd_data_plot_1.loc[(pd_data_plot_1['Market Cap / Total Current Assets'] < 20) &
                                         (pd_data_plot_1['Market Cap / Total Current Assets'] > 10)]
    pd_data_plot_1 = pd_data_plot_1.loc[(pd_data_plot_1['growth-1_B'] >= 0.05) & (pd_data_plot_1['growth-2_B'] >= 0.05)]
    pd_data_plot_1 = pd_data_plot_1.loc[(pd_data_plot_1['growth-1_C'] >= 0.05) & (pd_data_plot_1['growth-2_C'] >= 0.05)]
    pd_data_plot_1 = pd_data_plot_1.loc[(pd_data_plot_1['growth-1_D'] >= 0.05) & (pd_data_plot_1['growth-2_D'] >= 0.05)]

    fig, _ax = plt.subplots(2, 3, figsize=(18, 9))
    ax = fig.axes
    ax[0].hist(pd_data_plot_1['growth_A'], bins=50, histtype='step')
    ax[0].hist(pd_data_plot_0['growth_A'], bins=50, histtype='step')

    ax[1].hist(pd_data_plot_1['cap_growth'], bins=50, histtype='step')
    ax[1].hist(pd_data_plot_0['cap_growth'], bins=50, histtype='step')

    ax[2].plot(pd_data_plot_1.groupby('year').cap_growth.mean(), '.')
    ax2_t = ax[2].twinx()
    ax2_t.plot(pd_data_plot_1.groupby('year').size(), '.', color='red')

    ax[3].hist(pd_data_plot_1['Market Cap / Total Current Assets'], bins=50, histtype='step')
    ax[3].hist(pd_data_plot_0['Market Cap / Total Current Assets'], bins=50, histtype='step')

    x, y = pd_data_plot_1['Market Cap / Total Current Assets'], pd_data_plot_1['cap_growth']
    z = np.polyfit(x, y, 5)
    x_inter = np.linspace(x.min(), x.max(), 100)
    y_inter = np.poly1d(z)(x_inter)
    ax[4].plot(pd_data_plot_1['Market Cap / Total Current Assets'], pd_data_plot_1['cap_growth'], '.')
    ax[4].plot(x_inter, y_inter)

    print(f'{round(pd_data_plot_0.cap_growth.mean() * 100, 3)}%',
          f'{round(pd_data_plot_0.cap_growth.std() * 100, 3)}%', len(pd_data_plot_0))
    print(f'{round(pd_data_plot_1.cap_growth.mean() * 100, 3)}%',
          f'{round(pd_data_plot_0.cap_growth.std() * 100, 3)}%', len(pd_data_plot_1))

    result = 1
    pd_growth_rate_1 = pd_data_plot_1.groupby('year').cap_growth.mean()
    ax[5].hist(pd_growth_rate_1, histtype='step', bins=25)
    for i in pd_growth_rate_1:
        result *= (1+i)
    print(f'{round((result ** (1 / len(pd_growth_rate_1)) - 1) * 100, 2)}%', len(pd_growth_rate_1), round(result, 2), )


if __name__ == '__main__':

    self = StockAnalyze()
    try:
        temp = pd_sum.iloc[0]
    except:
        pd_sum = self.get_pd_sum()

    n_quarters_min = 12

    item_id_list = ['Total Current Assets', 'Total Revenue', 'Cash from Ops.',
                    'Gross Profit', 'Total Cash & ST Investments']
    item_id_include_list = item_id_list + ['Total Current Assets', 'Market Cap']
    ratio_col_name_new = 'MC_TCA'
    pd_sum = pd_sum.loc[(pd_sum.item_id.isin(item_id_include_list)) & (pd_sum.tq > 2005)]
    pd_mc_asset, ratio_col_name = self.get_stock_ratio(pd_sum, item_id_x='Total Current Assets',
                                                       item_id_y='Market Cap', col_val='value')
    pd_mc_asset = pd_mc_asset.rename(columns={ratio_col_name: ratio_col_name_new, 'time': 'year'})
    pd_mc_asset = pd_mc_asset[['exchange_ticker', 'year', ratio_col_name_new]]
    pd_mc_asset = pd_mc_asset.loc[(pd_mc_asset[ratio_col_name_new] > 5) & (pd_mc_asset[ratio_col_name_new] < 25)]
    exchange_ticker_list = list(pd_mc_asset.exchange_ticker.unique())

    pd_exchange_ticker = pd_sum.loc[pd_sum.exchange_ticker.isin(exchange_ticker_list)]
    pd_exchange_ticker = pd_exchange_ticker[['exchange_ticker', 'time']].drop_duplicates().groupby('exchange_ticker').size()
    pd_exchange_ticker = pd_exchange_ticker.loc[pd_exchange_ticker > n_quarters_min].rename('num').reset_index()
    exchange_ticker_list = list(pd_exchange_ticker.exchange_ticker)[:8]
    np.random.shuffle(exchange_ticker_list)
    pd_sum_select = pd_sum.loc[pd_sum.exchange_ticker.isin(exchange_ticker_list)]


    def get_pd_growth(pd_sum_select, n_year=2, mp_id=0, mp_queue=None):

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

                pd_stock_sec_sec = self.get_stock_growth(pd_stock_sec, item_id_list=item_id_list, n_year=n_year)
                pd_item_growth_sec_list.append(pd_stock_sec_sec)
            pd_item_growth_entry = pd.concat(pd_item_growth_sec_list)

            pd_item_growth_list.append(pd_item_growth_entry)
            if mp_queue is not None:
                mp_queue.put({mp_id: (i_ticker + 1) / len(exchange_ticker_list)})
            else:
                time_span = round(time.time() - time_start, 1)
                print(f'\rTime: {time_span} - Progress: {len(exchange_ticker_list)}/{i_ticker + 1}', end='')

        pd_item_growth = pd.concat(pd_item_growth_list)
        pd_item_growth = pd_item_growth.merge(pd_mc_asset, on=['exchange_ticker', 'year'], how='inner')

        dict_item_id = {'A': 'Total Current Assets', 'B': 'Cash from Ops.',
                        'C': 'Total Cash & ST Investments', 'D': 'Gross Profit'}
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
        pd_data = pd_data.merge(pd_item_growth[merge_cols+[ratio_col_name_new]].drop_duplicates(), on=merge_cols, how='inner')
        pd_data_growth = pd_data.copy()
        if mp_queue is not None:
            mp_queue.put(pd_data_growth)

        return pd_data_growth


    def mp_print_progress(mp_queue, time_start, dict_progress):
        if not mp_queue.empty():
            _result = mp_queue.get()
            if type(result) is dict:
                dict_progress['progress'].update(_result)
            else:
                dict_progress['result'].append(_result)
        time_span = round(time.time() - time_start, 1)
        progress = round(min(dict_progress['progress'].values()) * 100)
        print(f'\rTime: {time_span} s - Progress: {progress}%', end='')
        return dict_progress


    mp_queue = mp.Queue()
    n_process = 8
    n_year=2
    time_start = time.time()
    dict_progress = {'progress': {i: 0 for i in range(n_process)}, 'result': [], 'completed': [n_process, 0]}
    for i in range(n_process):
        n_total, n_sec = len(exchange_ticker_list), len(exchange_ticker_list) // n_process + 1
        exchange_ticker_list_sec = exchange_ticker_list[(i * n_sec):((i + 1) * n_sec)]
        pd_sum_sec = pd_sum_select.loc[pd_sum_select.exchange_ticker.isin(exchange_ticker_list_sec)]
        mp.Process(target=get_pd_growth, args=(pd_sum_sec, n_year, i, mp_queue)).start()

    while len(dict_progress['result']) != n_process:
        time.sleep(0.25)
        dict_progress = mp_print_progress(mp_queue, time_start, dict_progress)


    pd_data_growth = get_pd_growth(pd_sum_select, n_year=2, mp_queue=None)
    pd_data_growth.to_pickle(f'{DIR}/static/pd_data_grow_{common_func.date(0)}.pkl')


if __name__ == '__main__1':
    path_growth = max(glob.glob(f'{DIR}/static/pd_data_grow_*.pkl'))
    pd_data_growth = pd.read_pickle(path_growth)
    pd_data_growth = pd_data_growth.rename(columns={'year': 'time'})
    pd_data_plot_1 = pd_data_growth.loc[(pd_data['growth_A'] > 0.1) & (pd_data_growth['growth-1_A'] > 0.1)]
    pd_data_plot_0 = pd_data_plot_1.copy()
    pd_data_plot_1 = pd_data_plot_1.loc[(pd_data_plot_1[ratio_col_name_new] < 20) &
                                        (pd_data_plot_1[ratio_col_name_new] > 10)]
    pd_data_plot_1 = pd_data_plot_1.loc[(pd_data_plot_1['growth_B'] >= 0.05) & (pd_data_plot_1['growth-1_B'] >= 0.05)]
    pd_data_plot_1 = pd_data_plot_1.loc[(pd_data_plot_1['growth_C'] >= 0.05) & (pd_data_plot_1['growth-1_C'] >= 0.05)]
    pd_data_plot_1 = pd_data_plot_1.loc[(pd_data_plot_1['growth_D'] >= 0.05) & (pd_data_plot_1['growth-1_D'] >= 0.05)]

    dict_in_stock = {'time': [], 'exchange_ticker': []}
    for time in sorted(pd_data_plot_1.time.unique()):
        pd_stock_buy = pd_data_plot_1.loc[pd_data_plot_1.time == time]
        exchange_ticker_in_stock_list = list(pd_stock_buy.exchange_ticker)
        dict_in_stock['exchange_ticker'] += exchange_ticker_in_stock_list
        dict_in_stock['time'] += [time] * len(exchange_ticker_in_stock_list)
    pd_stock_bu = pd.DataFrame(dict_in_stock)
    pd_stock_bu['symbol'] = pd_stock_bu['exchange_ticker'].str.split(':').str[-1]
    pd_stock_bu['status'] = 1
    pd_stock_bu['time'] = (pd.to_datetime(pd_stock_bu['time']) + pd.to_timedelta('1 day')).astype(str).str[:10]

    hold_day_min = 210
    pd_stock_se_list = []
    symbols = sorted(pd_stock_bu.symbol.unique())
    symbols = ['CMG']
    pd_stock_bu['time'] = pd.to_datetime(pd_stock_bu['time'])
    dates = sorted(pd_stock_bu.time.unique())
    dict_dates_convert = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    # Get the dates that sells stocks
    for i_symbol, symbol in zip(range(len(symbols)), symbols):
        pd_symbol = pd_stock_bu.loc[pd_stock_bu.symbol == symbol]
        if len(pd_symbol) > 1:
            pd_time_diff = pd_symbol.time.diff().reset_index()
            pd_time_diff['time'] = list(pd_time_diff['time'])[1:] + [np.nan]
            index_selling = pd_time_diff.loc[pd_time_diff.time.dt.days >= hold_day_min]['index']
            pd_selling = pd_symbol.loc[index_selling].copy()
        else:
            pd_selling = pd_symbol.copy()
        pd_selling['time'] = (pd_selling.time + pd.to_timedelta(f'{hold_day_min} days'))
        pd_selling['status'] = 0
        pd_stock_se_list.append(pd_selling)
    pd_stock_se = pd.concat(pd_stock_se_list)


    # Add price information to the buying stocks
    pd_stock_act = pd.concat([pd_stock_bu, pd_stock_se])
    pd_stock_act['time'] = pd_stock_act['time'].astype(str).str[:10]
    pd_stock_pos_list = []
    symbols = sorted(pd_stock_act.symbol.unique())
    time_start = time.time()
    for i_symbol, symbol in zip(range(len(symbols)), symbols):
        pd_symbol = pd_stock_act.loc[pd_stock_act.symbol == symbol]
        dates = list(pd_symbol.time)
        pd_price = stock_price.get_price_dates(symbol, dates)
        pd_price['time'] = pd_price['time_request']
        pd_price = pd_price[['time', 'close']].rename(columns={'close': 'price'})
        pd_symbol = pd_symbol.merge(pd_price, on='time', how='inner')
        pd_stock_pos_list.append(pd_symbol)
        time_span = round(time.time() - time_start, 1)
        print(f'\rTime: {time_span} - Progress: {len(symbols)}/{i_symbol + 1}', end='')

    pd_stock_act = pd.concat(pd_stock_pos_list)

    dates = list(pd_stock_act.time.unique())
    for date in dates:
        a=1