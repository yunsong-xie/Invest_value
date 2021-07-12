
import re, os, sys
import numpy as np
import pandas as pd
import datetime
from bs4 import BeautifulSoup
import time
import urllib.request
from termcolor import colored
import glob, xlrd


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


def get_dict_symbol(workbook):
    sheet_HC = workbook.sheet_by_name('Historical Capitalization')
    HC_symbol = search_1st_col(sheet_HC, 'Financials', strict=False)
    exchange_ticker = re.findall('[A-Za-z]*:[A-Z]*', sheet_HC.cell(HC_symbol, 0).value)[0]
    exchange, symbol = exchange_ticker.split(':')
    dict_symbol = {'symbol': symbol, 'exchange': exchange, 'exchange_ticker': exchange_ticker}
    return dict_symbol

def get_template(workbook):
    sheet_HC = workbook.sheet_by_name('Income Statement')
    row_ind=search_1st_col(sheet_HC, 'Template:', col_index=1, strict=True)
    template=sheet_HC.cell(row_ind, 2).value
    return template


# dict_sum={'exchange': [], 'symbol':[], 'exchange_ticker':[], 'sheet_id':[], 'item':[], 'value':[], 'time':[]}


def get_row_value(pd_sum_input, workbook, sheet_id_list='Historical Capitalization',
                  item_id_list='Share Price', time_list=[]):
    """
    This function output pd dataframe there are two cases:
    1. time_list == []: time_list initiation, query "Share Price" row, get the time_list that has available share Price
    2. time_list != []: Get the row with given name "Net Income", "Revenue", etc. Only select the entries
                        that is within the "time_list" input
    :param pd_sum_input:    Current available pd_sum
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
        sheet_id_list=[sheet_id_list]

    if len(time_list) == 0:
        label_base = 1
        sheet_id_list = ['Historical Capitalization']
        item_id_list = ['Share Price']
    else:
        label_base = 0

    label_success=0
    count_parse=-1

    while (label_success==0)&(count_parse<len(sheet_id_list)-1):
        count_parse+=1
        sheet_id=sheet_id_list[count_parse]
        item_id=item_id_list[count_parse]

        sheet_HC = workbook.sheet_by_name(sheet_id)
        dict_sum = {'sheet_id': [], 'item_id': [], 'value': [], 'time': []}
        time_row_ind = search_1st_col(sheet_HC, dict_time_name[sheet_id])
        value_list_ori = sheet_HC.row_values(time_row_ind)
        pattern = '[A-Za-z]*-[0-9]*-[0-9]*'
        time_list_temp = [re.search(pattern, i) for i in value_list_ori]

        item_row_num = search_1st_col(sheet_HC, item_id)
        if item_row_num != -1:
            # The row does exist
            HC_share_price = sheet_HC.row_values(search_1st_col(sheet_HC, item_id))
            if label_base == 1:
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

        time_list = [dict_time[i] for i in list(dict_share_price.keys())]
        item_id_record=item_id_list[0]
        if type(item_id_record) is list:
            item_id_record=item_id_record[0]
        for i in list(dict_share_price.keys()):
            if dict_share_price[i]!='-':
                dict_sum['sheet_id'].append(sheet_id)
                dict_sum['item_id'].append(item_id_record.strip())
                dict_sum['value'].append(dict_share_price[i])
                dict_sum['time'].append(dict_time[i])

        pd_sum_new=pd.DataFrame(dict_sum)
        if len(pd_sum_new)>0:
            label_success=1


    pd_sum_output = pd.concat([pd_sum_input, pd_sum_new])

    if label_base == 1:
        return pd_sum_output, time_list
    else:
        return pd_sum_output


dict_time_name = {'Income Statement': 'For the Fiscal Period Ending',
                  'Balance Sheet': 'Balance Sheet as of:',
                  'Cash Flow': 'For the Fiscal Period Ending',
                  'Historical Capitalization': 'Balance Sheet as of:',
                  'Ratios': 'For the Fiscal Period Ending',
                  'Segments': 'For the Fiscal Period Ending', }

parse_data = [[['Income Statement'], ['Total Revenue']],
              [['Income Statement'], ['Operating Income']],
              [['Balance Sheet'], ['Long-Term Debt']],
              [['Balance Sheet'], 'Long-Term Leases'],
              [['Cash Flow', 'Income Statement'], ['Net Income', 'Net Income']],
              [['Cash Flow'], ['Cash from Ops.']],
              [['Income Statement'], ['Total Assets']],
              [['Balance Sheet'], ['Total Current Assets']],
              [['Balance Sheet'], ['Total Current Liabilities']],
              [['Balance Sheet'], ['Total Equity']],
              [['Income Statement'], ['Revenue']],
              [['Balance Sheet'], ['Retained Earnings']],
              [['Income Statement'], ['EBIT']],
              [['Historical Capitalization'], ['Market Capitalization']],
              [['Balance Sheet'], ['Total Liabilities']],
              [['Balance Sheet'], ['Accounts Receivable']],
              [['Income Statement'], ['Gross Profit']],
              [['Balance Sheet'], ['Net Property, Plant & Equipment']],
              [['Balance Sheet'], ['Accumulated Depreciation']],
              [['Income Statement'], ['Selling General & Admin Exp.']],
              [['Income Statement'], ['Other Operating Expense']],
              [['Cash Flow'], ['Depreciation & Amort., Total']],
              [['Income Statement'], ['EBITDA']],
              [['Income Statement'], ['Diluted EPS']],
              [['Balance Sheet'], ['Total Cash & ST Investments']],
              ]

class MyFilter(object):
    def __init__(self, mylogfile=sys.stdout):
        self.f = mylogfile
    def write(self, data):
        if "WARNING *** OLE2 inconsistency" not in data:
            self.f.write(data)
log = open("the_log_file.txt", "w")
log_filter = MyFilter(log)

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

folder_path= '/Invest_value/capitalIQ/Financial_reports'

path_pd_ciq_id= 'static/csv/stock_list/pd_ciq_id.pickle'
path_parsed_result= 'static/csv/stock_list/pd_sum.pickle'
pd_ciq_id=pd.read_pickle(path_pd_ciq_id)
pd_ciq_id_company=pd_ciq_id.loc[pd_ciq_id.company_type=='company'].copy()
pd_ciq_id_company['file_keyword']=pd_ciq_id_company['exchange_ticker'].str.replace(':', ' ')
file_keyword_company_list=list(pd_ciq_id_company['file_keyword'])

pd_sum_list=[]

file_keyword_company_list=file_keyword_company_list

count_file=0
time_start=time.time()

file_keyword_company_list=file_keyword_company_list

if 1==1:
    count_valid=0
    for file_keyword_company in file_keyword_company_list:
        filename = glob.glob(folder_path + '\* '+file_keyword_company+' *.xls')
        if len(filename)==0:
            continue
        else:
            filename = filename[0]

        count_file+=1
        workbook=xlrd.open_workbook(filename, logfile=log_filter)

        sheetname_list=workbook.sheet_names()

        pd_sum_symbol=pd.DataFrame({'sheet_id':[], 'item_id':[], 'value':[], 'time':[]})

        try:
            pd_sum_symbol, time_list=get_row_value(pd_sum_symbol, workbook)


            for parse_data_entry in  parse_data:
                sheet_id_list, item_id_list = parse_data_entry
                pd_sum_symbol=get_row_value(pd_sum_symbol, workbook, sheet_id_list = sheet_id_list,
                                            item_id_list=item_id_list, time_list=time_list)
            count_valid+=1
        except:
            continue

        template = get_template(workbook)
        pd_sum_symbol['template']=template
        dict_symbol=get_dict_symbol(workbook)
        for key in dict_symbol:
            pd_sum_symbol[key] = dict_symbol[key]

        workbook.release_resources()
        pd_sum_list.append(pd_sum_symbol)
        time_span=round(time.time()-time_start, 1)
        print('\rTime: '+str(time_span)+' - Complete parse file '+str(count_file)+'/'+str(len(file_keyword_company_list))+
              ' - Valid data entries: '+str(count_valid), end='')

    log.close()
    pd_sum=pd.concat(pd_sum_list)

    pd_sum.to_pickle(path_parsed_result)



if 1==0:
    #file_keyword_company = 'NasdaqGS ZION'
    file_keyword_company = 'NYSE AAN'
    filename = glob.glob(folder_path + '\* ' + file_keyword_company + ' *.xls')[0]
    workbook = xlrd.open_workbook(filename)
    pd_sum_symbol = pd.DataFrame({'sheet_id': [], 'item_id': [], 'value': [], 'time': []})
    parse_data_entry = ['Balance Sheet', 'Accumulated Depreciation']
    sheet_id_entry, item_id_entry = parse_data_entry[0], parse_data_entry[1:]
    _, time_list = get_row_value(pd_sum_symbol, workbook)
    get_row_value(pd_sum_symbol, workbook, sheet_id=sheet_id_entry,
                  item_id=item_id_entry, time_list=time_list)
