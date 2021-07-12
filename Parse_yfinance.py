__author__ = 'Yunsong Xie'
__email__ = 'xiefinance00@gmail.com'
__company__ = 'Xie Finance LLC'

import os, glob, time, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import lib as common_func

pd.set_option('display.max_column', 60)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 10000)

DIR = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")
DICT_ITEMS = {'balance_sheet': ['TotalAssets', 'CurrentAssets', 'CashAndCashEquivalents', 'TotalLiabilitiesNetMinorityInterest',
                                'CurrentLiabilities', 'TangibleBookValue', 'ShareIssued'],
              'cach_flow': ['OperatingCashFlow', 'FreeCashFlow'],
              'financials': ['TotalRevenue', 'OperatingRevenue', 'GrossProfit', 'OperatingExpense',
                             'OperatingIncome', 'NetIncome']}
YF_ITEMS = []
for _items in list(DICT_ITEMS.values()):
    YF_ITEMS += _items


def add_price_info(symbol, pd_fr_symbol):
    """
    Add price information (adj-close and adj-low) to financial info database. Column names of these two properties are:
    price_close, and price_low.
    Args:
        symbol (str): Stock symbol
        pd_fr_symbol (pandas.dataframe): financial report in from yahoo. If should at least contain column "date"

    Returns:
        (pandas.dataframe): New financial report in from yahoo, with the close and low price properties added. If no
            price info is found for that symbol, return None.
    """
    pd_price = stock_price.get_price_range(symbol)
    if pd_price is None:
        return None
    pd_price = pd_price.sort_values(by='time', ascending=False)
    pd_price['time'] = pd_price.time.astype(str).str[:10]
    pd_fr_symbol = pd_fr_symbol.loc[pd_fr_symbol.date >= pd_price['time'].min()].copy()

    pd_earning = stock_price.get_earning_dates(symbol)
    pd_earning = pd_earning.loc[pd_earning.time != 'NA']
    if (len(pd_earning) == 0):
        return None
    elif len(pd_fr_symbol) == 0:
        return None
    elif pd_fr_symbol.date.max() > pd_earning.time.max():
        return None

    time_earning_earliest = common_func.unix2date(common_func.date2unix(pd_earning.time.min()) - 3600 * 24 - 90)[:10]
    pd_fr_symbol = pd_fr_symbol.loc[pd_fr_symbol.date >= time_earning_earliest]


    pd_earning = pd_earning.loc[pd_earning.time >= pd_price['time'].min()].sort_values(by='time', ascending=False)
    earning_dates = np.asarray(sorted(pd_earning.time, reverse=True))
    earning_month = np.asarray([i[:7] for i in earning_dates])
    dict_price, report_dates = {'price_close': [], 'price_low': [], 'date_earn': []}, []
    for _date in list(pd_fr_symbol.date):
        # Some companry, AZO for example, has the yahoo finance reported date showing later date (end of the month) comparing
        # to the actual reporting date. For this reason, just compare the month.
        report_date = earning_dates[earning_month >= _date[:7]][-1]
        report_date_after = common_func.unix2date(common_func.date2unix(report_date) + 3600 * 24 * 1)[:10]
        pd_price_after_earn = pd_price.loc[pd_price.time >= report_date_after].iloc[-1]
        price_close, price_close_adj, price_low, _time = pd_price_after_earn[['close', 'adjclose', 'low', 'time']]
        price_low_adj = price_low / price_close * price_close_adj
        dict_price['price_close'].append(price_close_adj)
        dict_price['price_low'].append(price_low_adj)
        dict_price['date_earn'].append(_time)

    for key in dict_price:
        pd_fr_symbol[key] = dict_price[key]
    return pd_fr_symbol


def get_yf_data(dir_csv=None, force_reload=False):
    """
    Parse the csv financial data from yahoo finance.
    If the organization does not include any of the pre-determined several columns do not include this organization.

    Args:
        dir_csv (str): directory of the new yahoo finance financial report in form of csv. if nothing is provided,
            just output the previously obtained dict data (default: None)

    Returns:
        (dict): A dict that key is the symbol, value is the financial data in pandas.dataframe form
    """
    pkl_filename = f'{DIR}/static/analysis/yf_data/dict_yf.pkl'
    dir_fr_source = f'{DIR}/static/Financial_reports/YF_FR'


    fr_files = []
    if dir_csv is not None:
        fr_files += glob.glob(f'{dir_csv}/*.csv')

    if os.path.isfile(pkl_filename) & (not force_reload):
        with open(pkl_filename, 'rb') as f:
            dict_yf = pickle.load(f)
        return dict_yf
    else:
        fr_files += glob.glob(f'{dir_fr_source}/*.csv')
        dict_yf = {}
    dict_fr_file = {os.path.basename(i)[:-4]: i for i in fr_files}

    symbols = sorted(dict_fr_file.keys())
    dic_fr_sum = {'symbol': [], 'miss': []}
    time_start = time.time()

    for i_symbol, symbol in zip(range(len(symbols)), symbols):
        fr_file = dict_fr_file[symbol]
        pd_fr_symbol_ori = pd.read_csv(fr_file)

        pd_fr_symbol_select = pd_fr_symbol_ori.loc[pd_fr_symbol_ori.item.isin(YF_ITEMS)]
        items_symbol = list(pd_fr_symbol_select.item)
        if len(items_symbol) == len(YF_ITEMS):

            cols = [i for i in pd_fr_symbol_select.keys() if i not in ['section', 'item']]
            pd_fr_symbol = pd_fr_symbol_select[cols].T
            columns = list(pd_fr_symbol_select.item)
            pd_fr_symbol.columns = list(columns)

            pd_fr_symbol = pd_fr_symbol[items_symbol]
            pd_fr_symbol = pd_fr_symbol.loc[~pd_fr_symbol.TangibleBookValue.isna()].reset_index()
            pd_fr_symbol['symbol'] = symbol
            pd_fr_symbol = pd_fr_symbol.rename(columns={'index': 'date'})
            keys = ['date', 'symbol'] + [i for i in pd_fr_symbol.keys() if i not in ['date', 'symbol']]
            pd_fr_symbol = pd_fr_symbol[keys]

            pd_fr_symbol = add_price_info(symbol, pd_fr_symbol)
            if pd_fr_symbol is None:
                if symbol in dict_yf:
                    del dict_yf[symbol]
                continue

            dict_yf[symbol] = pd_fr_symbol

            for item in [_ for _ in YF_ITEMS if _ not in items_symbol]:
                dic_fr_sum['symbol'].append(symbol)
                dic_fr_sum['miss'].append(item)
        if i_symbol % 20 == 0:
            time_span = round(time.time() - time_start, 1)
            print(f'\rTime: {time_span} s - {i_symbol + 1}/{len(symbols)}', end='')
    pd_fr_sum = pd.DataFrame(dic_fr_sum)
    print()

    with open(pkl_filename, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(dict_yf, f, pickle.HIGHEST_PROTOCOL)
    print(f'Output to {pkl_filename}')

    return dict_yf


stock_price = common_func.StockPrice()


if __name__ == '__main__0':
    try:
        symbols = list(dict_yf.keys())
    except:
        dict_yf = get_yf_data(dir_csv=None, force_reload=False)
        symbols = list(dict_yf.keys())
    symbols = sorted(dict_yf.keys())
    stock_price.update_price_symbol(symbols)
    pd.DataFrame({'symbol': symbols}).to_csv(f'{DIR}/static/Financial_reports/pd_symbol.csv', index=False)


def get_yf_year_data(force_reload=False, n_year=3):
    pkl_filename = f'{DIR}/static/analysis/yf_data/dict_yf_year.pkl'
    is_reload = (not os.path.isfile(pkl_filename)) | force_reload
    if not is_reload:
        with open(pkl_filename, 'rb') as f:
            dict_yf_year = pickle.load(f)
        return dict_yf_year
    else:
        try:
            symbols = list(dict_yf.keys())
        except:
            dict_yf = get_yf_data(dir_csv=None)
            symbols = list(dict_yf.keys())
        symbols = list(dict_yf.keys())
        dict_yf_year, time_start = {}, time.time()

        for i_symbol, symbol in zip(range(len(symbols)), symbols):
            pd_fr_symbol = dict_yf[symbol]
            pd_fr_symbol_growth = pd_fr_symbol[[_ for _ in pd_fr_symbol.keys() if _ not in YF_ITEMS]].copy()
            pd_fr_symbol_growth = pd_fr_symbol_growth.iloc[:len(pd_fr_symbol_growth) - n_year * 4 + 1].copy()
            if len(pd_fr_symbol) >= (n_year * 4):
                for item in YF_ITEMS:
                    data = np.asarray(pd_fr_symbol[item]).reshape(len(pd_fr_symbol), 1)
                    data1 = np.concatenate((data[:-3], data[1:-2], data[2:-1], data[3:]), axis=1)
                    data_growth = data1.mean(axis=1)
                    for i_year in range(n_year):
                        growth_item = f'{item}-{i_year}'
                        pd_fr_symbol_growth[growth_item] = data_growth[(i_year * 4):(len(data) - (n_year - i_year) * 4 + 1)]
                dict_yf_year[symbol] = pd_fr_symbol_growth
                time_span = round(time.time() - time_start, 1)
                print(f'\rTime: {time_span} s - {i_symbol + 1}/{len(symbols)}', end='')
        print()
        with open(pkl_filename, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(dict_yf_year, f, pickle.HIGHEST_PROTOCOL)
        print(f'Output to {pkl_filename}')
        return dict_yf_year


if __name__ == '__main__0':
    try:
        symbols = list(dict_yf_year.keys())
    except:
        dict_yf_year = get_yf_year_data(force_reload=False)
        symbols = list(dict_yf_year.keys())


def get_pd_yf_growth(force_reload=False):
    pkl_filename = f'{DIR}/static/analysis/yf_data/pd_yf_growthpkl'
    is_reload = (not os.path.isfile(pkl_filename)) | force_reload
    if not is_reload:
        pd_yf_growth = pd.read_pickle(pkl_filename)
        return pd_yf_growth
    else:
        try:
            symbols = list(dict_yf_year.keys())
        except:
            dict_yf_year = get_yf_year_data(force_reload=False)
            symbols = list(dict_yf_year.keys())

        symbols = sorted(dict_yf_year.keys())
        time_start = time.time()
        items_exclude = ['ShareIssued']
        pd_yf_grow_list = []
        for i_symbol, symbol in zip(range(len(symbols)), symbols):
            pd_yf_year = dict_yf_year[symbol]
            pd_yf_year['MarketCap'] = pd_yf_year['ShareIssued-0'] * pd_yf_year['price_close']
            items = [i for i in pd_yf_year.keys() if i.split('-')[0] not in YF_ITEMS] + ['ShareIssued-0']
            pd_yf_growth_symbol = pd_yf_year[items].rename(columns={'ShareIssued-0': 'ShareIssued'}).copy()
            n_year = max([int(i.split('-')[-1]) for i in pd_yf_year.keys() if i.split('-')[0] in YF_ITEMS])
            yf_items_filtered = [i for i in YF_ITEMS if i not in items_exclude]
            for item in yf_items_filtered:
                pd_yf_growth_symbol[f'{item}-0'] = pd_yf_year[f'{item}-0']
                for i_year in range(n_year):
                    pd_yf_growth_symbol[f'{item}-diff-{i_year}'] = pd_yf_year[f'{item}-{i_year}'] - pd_yf_year[f'{item}-{i_year + 1}']

            pd_yf_grow_list.append(pd_yf_growth_symbol)
            time_span = round(time.time() - time_start, 1)
            print(f'\rGetting pd_yf_growth - time {time_span} s - progress {i_symbol + 1}/{len(symbols)}', end='')
        print()
        pd_yf_growth = pd.concat(pd_yf_grow_list)
        pd_yf_growth.to_pickle(pkl_filename)
    return pd_yf_growth


def filter_yf_growth(pd_yf_growth, dict_filter):
    if dict_filter is not None:
        if '>' in dict_filter:
            for item_list in dict_filter['>']:
                if len(item_list) == 2:
                    item_1, item_2 = item_list
                    if type(item_2) is str:
                        pd_yf_growth = pd_yf_growth.loc[(pd_yf_growth[item_1] > pd_yf_growth[item_2])]
                    else:
                        pd_yf_growth = pd_yf_growth.loc[(pd_yf_growth[item_1] > item_2)]
                else:
                    item_1, item_2, item_3 = item_list
                    pd_yf_growth = pd_yf_growth.loc[(pd_yf_growth[item_1] / pd_yf_growth[item_2]) > item_3]
        if '<' in dict_filter:
            for item_list in dict_filter['<']:
                if len(item_list) == 2:
                    item_1, item_2 = item_list
                    if type(item_2) is str:
                        pd_yf_growth = pd_yf_growth.loc[(pd_yf_growth[item_1] < pd_yf_growth[item_2])]
                    else:
                        pd_yf_growth = pd_yf_growth.loc[(pd_yf_growth[item_1] < item_2)]
                else:
                    item_1, item_2, item_3 = item_list
                    pd_yf_growth = pd_yf_growth.loc[(pd_yf_growth[item_1] / pd_yf_growth[item_2]) < item_3]
    return pd_yf_growth


def get_hold_act(pd_yf_growth, dict_filter_1, dict_filter_2, hold_day_min=370):
    pd_yf_growth = pd_yf_growth.copy()
    items_exclude = ['OperatingExpense', 'OperatingIncome', 'CurrentLiabilities', 'TotalLiabilitiesNetMinorityInterest']
    items = [i for i in pd_yf_growth.keys() if i.split('-')[0] not in items_exclude]
    pd_yf_growth = pd_yf_growth[items]

    pd_yf_growth_1 = filter_yf_growth(pd_yf_growth, dict_filter_1)
    pd_yf_growth_start = filter_yf_growth(pd_yf_growth_1, dict_filter_2['start'])
    pd_yf_growth_restart = filter_yf_growth(pd_yf_growth_1, dict_filter_2['restart']).copy()
    pd_yf_growth_restart['act'] = 'restart'

    merge_cols = ['symbol', 'date', 'date_earn']
    pd_hold = pd_yf_growth_restart[merge_cols + ['act']].merge(pd_yf_growth_start, on=merge_cols, how='outer')
    pd_hold_filter = pd_hold.groupby(['symbol', 'date_earn'])['date'].max().reset_index()
    pd_hold = pd_hold.merge(pd_hold_filter, on=merge_cols, how='inner')
    pd_hold['act'] = pd_hold['act'].fillna('start')
    pd_hold = pd_hold.sort_values(by=['symbol', 'date'])
    time_start = time.time()

    symbols = sorted(pd_hold.symbol.unique())
    pd_hold_act_list = []

    for i_symbol, symbol in zip(range(len(symbols)), symbols):

        pd_hold_symbol = pd_hold.loc[pd_hold.symbol == symbol]
        pd_hold_symbol_act_list = []
        date_earn_pre, date_sold, is_sold = '1997-01-01', '1997-01-01', False

        for i in range(len(pd_hold_symbol)):
            date_earn, act = pd_hold_symbol.iloc[i][['date_earn', 'act']]
            if i == 0:
                pd_hold_symbol_act_list.append(pd_hold_symbol.iloc[[i]])
                date_earn_pre = date_earn
                if len(pd_hold_symbol) == 1:
                    date_last = common_func.unix2date(common_func.date2unix(date_earn_pre) + 3600 * 24 * hold_day_min)[:10]
                    if date_last < common_func.date(0):
                        pd_sell = pd.DataFrame({'symbol': [symbol], 'date_earn': [date_last], 'act': ['sell']})
                    else:
                        pd_sell = pd.DataFrame({'symbol': [symbol], 'date_earn': [common_func.date(0)], 'act': ['end']})
                    pd_hold_symbol_act_list.append(pd_sell)
            else:
                date_last = common_func.unix2date(common_func.date2unix(date_earn_pre) + 3600 * 24 * hold_day_min)[:10]

                if date_earn > date_last:
                    if not is_sold:
                        pd_sell = pd.DataFrame({'symbol': [symbol], 'date_earn': [date_last], 'act': ['sell']})
                        pd_hold_symbol_act_list.append(pd_sell)
                        date_sold = date_last
                        is_sold = True

                if (date_earn < date_last) | (act == 'restart'):
                    pd_hold_symbol_act_list.append(pd_hold_symbol.iloc[[i]])
                    date_earn_pre = date_earn
                    if act == 'restart':
                        is_sold = False

                if (i == (len(pd_hold_symbol) - 1)) & (not is_sold):
                    date_last = common_func.unix2date(common_func.date2unix(date_earn_pre) + 3600 * 24 * hold_day_min)[:10]
                    if date_last < common_func.date(0):
                        pd_sell = pd.DataFrame({'symbol': [symbol], 'date_earn': [date_last], 'act': ['sell']})
                    else:
                        pd_sell = pd.DataFrame({'symbol': [symbol], 'date_earn': [common_func.date(0)], 'act': ['end']})
                    pd_hold_symbol_act_list.append(pd_sell)

        pd_hold_symbol_act = pd.concat(pd_hold_symbol_act_list)

        def _add_price_info(pd_hold_symbol_act, act_key, pd_price):
            pd_price['adjlow'] = pd_price['low'] / pd_price['close'] * pd_price['adjclose']
            pd_hold_symbol_act.loc[pd_hold_symbol_act.act == act_key, 'price_close'] = list(pd_price['adjclose'])
            pd_hold_symbol_act.loc[pd_hold_symbol_act.act == act_key, 'price_low'] = list(pd_price['adjlow'])
            pd_hold_symbol_act.loc[pd_hold_symbol_act.act == act_key, 'date_earn'] = list(pd_price['time'].str[:10])
            return pd_hold_symbol_act

        pd_hold_symbol_act_sell = pd_hold_symbol_act.loc[pd_hold_symbol_act.act == 'sell']
        if len(pd_hold_symbol_act_sell) > 0:
            pd_price = stock_price.get_price_dates(symbol, list(pd_hold_symbol_act_sell.date_earn))
            pd_hold_symbol_act = _add_price_info(pd_hold_symbol_act, 'sell', pd_price)

        pd_hold_symbol_act_end = pd_hold_symbol_act.loc[pd_hold_symbol_act.act == 'end']
        if len(pd_hold_symbol_act_end) > 0:
            pd_price = stock_price.get_price_range(symbol, date_start=common_func.date(-14), date_end=common_func.date(0),
                                                   time_type='date')
            pd_price = pd_price.iloc[[-1]].copy()
            pd_hold_symbol_act = _add_price_info(pd_hold_symbol_act, 'end', pd_price)

        time_span = round(time.time() - time_start, 1)
        print(f'\rGetting pd_yf_growth - time {time_span} s - progress {i_symbol + 1}/{len(symbols)}', end='')
        pd_hold_act_list.append(pd_hold_symbol_act)
    print()
    pd_hold_act = pd.concat(pd_hold_act_list).copy()
    pd_hold_act['PB'] = pd_hold_act['MarketCap'] / pd_hold_act['TangibleBookValue-0']
    keys_front = [i for i in pd_hold_act.keys() if not (('0' in i) | ('1' in i))]
    keys_back = [i for i in pd_hold_act.keys() if i not in keys_front]
    pd_hold_act = pd_hold_act[keys_front + keys_back]
    return pd_hold_act


def bu_se_2_profit_fine(dict_pd_stock_bs, pd_stock_bs, n_stock_max=5):

    pd_stock_bs_recal_list, pd_stock_bs_ori_list = [], []

    for symbol in dict_pd_stock_bs:
        pd_stock_bs_symbol_ori = dict_pd_stock_bs[symbol].copy()
        pd_stock_bs_ori_list.append(pd_stock_bs_symbol_ori)

    pd_stock_bs_ori = pd.concat(pd_stock_bs_ori_list)
    pd_stock_bs_select = pd_stock_bs_ori

    pd_stock_act_sim = pd_stock_bs_select[['time', 'symbol', 'price_close', 'weight']].drop_duplicates()
    pd_stock_act_sim = pd_stock_act_sim.set_index(['time', 'symbol']).T
    dict_stock_act_ori = pd_stock_act_sim.to_dict()
    dict_stock_act = {}
    dates, symbols = sorted(pd_stock_bs['time'].unique()), list(pd_stock_bs['symbol'].unique())
    for date in dates:
        for symbol in symbols:
            if (date, symbol) in dict_stock_act_ori:
                if date in dict_stock_act:
                    dict_stock_act[date].update({symbol: dict_stock_act_ori[(date, symbol)]})
                else:
                    dict_stock_act[date] = {symbol: dict_stock_act_ori[(date, symbol)]}

    time_start = time.time()
    dict_holding = {'cash': 100}
    dict_reward = {'asset': [], 'date': [], 'n_stock': [], 'hold_num': [], 'hold_val': []}
    dates = sorted(dict_stock_act.keys())
    # dates = ['2009-11-06', '2009-11-09']
    for i_date, date in zip(range(len(dates)), dates):
        dict_act_ori = dict_stock_act[date]
        stock_value = sum([dict_holding[i] * dict_act_ori[i]['price_close'] for i in dict_holding
                           if (i != 'cash') & (dict_holding[i] > 0)])
        asset = stock_value + dict_holding['cash']
        dict_act_keys = sorted(list(dict_act_ori.keys()), key=lambda x: -dict_act_ori[x]['weight'])[:n_stock_max]
        dict_act = {i: dict_act_ori[i] for i in dict_act_keys}

        weight_total = sum([dict_act[i]['weight'] for i in dict_act])
        if weight_total > 0:
            weight_unit = asset / weight_total
            for symbol in dict_holding:
                dict_holding[symbol] = 0
            for symbol in dict_act:
                price = dict_act[symbol]['price_close']
                weight = dict_act[symbol]['weight']
                dict_holding[symbol] = weight_unit * weight / price
            dict_holding['cash'] = 0
        else:
            for symbol in dict_holding:
                dict_holding[symbol] = 0
            dict_holding['cash'] = asset
        dict_holding_num = {i: dict_holding[i] for i in dict_holding if (i != 'cash') & (dict_holding[i] > 0)}
        dict_holding_val = {i: (dict_holding[i] * dict_act[i]['price_close']) for i in dict_holding
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


if __name__ == '__main__0':
    try:
        symbols = sorted(pd_yf_growth.symbol.unique())
    except:
        pd_yf_growth = get_pd_yf_growth(force_reload=False)
        symbols = sorted(pd_yf_growth.symbol.unique())

    DICT_ITEMS = {'balance_sheet': ['TotalAssets', 'CurrentAssets', 'CashAndCashEquivalents', 'TotalLiabilitiesNetMinorityInterest',
                                    'CurrentLiabilities', 'TangibleBookValue', 'ShareIssued'],
                  'cach_flow': ['OperatingCashFlow', 'FreeCashFlow'],
                  'financials': ['TotalRevenue', 'OperatingRevenue', 'GrossProfit', 'OperatingExpense',
                                 'OperatingIncome', 'NetIncome']}


    dict_filter_1 = {'>': [['TotalAssets-diff-0', 0], ['TotalAssets-diff-1', 0],
                           ['CurrentAssets-diff-0', 0], ['CurrentAssets-diff-1', 0],
                           ['CashAndCashEquivalents-diff-0', 0], ['CashAndCashEquivalents-diff-1', 0],
                           ['TangibleBookValue-diff-0', 0], ['TangibleBookValue-diff-1', 0],
                           ['OperatingCashFlow-diff-0', 0], ['OperatingCashFlow-diff-1', 0],
                           ['FreeCashFlow-diff-0', 0], ['FreeCashFlow-diff-1', 0],
                           ['GrossProfit-diff-0', 0], ['GrossProfit-diff-1', 0],
                           ['OperatingRevenue-diff-0', 0], ['OperatingRevenue-diff-1', 0],
                           ['TangibleBookValue-0', 0],
                           ['MarketCap', 'TangibleBookValue-0', 5],
                           ],
                     '<': [['MarketCap', 'CurrentAssets-0', 50],
                           ['MarketCap', 'TangibleBookValue-0', 50]]}

    dict_filter_2 = {'start': {'>': [['TangibleBookValue-diff-0', 'TangibleBookValue-0', 0.3],
                                     ['CurrentAssets-diff-0', 'CurrentAssets-0', 0.2],
                                     ]},
                     'restart': {'>': [['TangibleBookValue-diff-0', 'TangibleBookValue-0', 10.4],
                                       ['FreeCashFlow-diff-0', 'FreeCashFlow-0', 0.2],
                                       ['CurrentAssets-diff-0', 'CurrentAssets-0', 0.2, ]]}}

    is_update = True
    pd_yf_growth_1 = pd_yf_growth
    for key in ['TangibleBookValue', 'GrossProfit', 'OperatingCashFlow', 'CashAndCashEquivalents', 'FreeCashFlow']:
        pd_yf_growth_1 = pd_yf_growth_1.loc[(pd_yf_growth_1[f'{key}-0'] - pd_yf_growth_1[f'{key}-diff-0'] > 0)]
        pd_yf_growth_1 = pd_yf_growth_1.loc[(pd_yf_growth_1[f'{key}-0'] - pd_yf_growth_1[f'{key}-diff-0'] - pd_yf_growth_1[f'{key}-diff-0'] > 0)]


    pd_hold_act = get_hold_act(pd_yf_growth_1, dict_filter_1, dict_filter_2, hold_day_min=370)
    #pd_yf_growth_temp = pd_yf_growth.loc[pd_yf_growth.symbol == symbol]
    #pd_hold_act_temp = pd_hold_act.loc[pd_hold_act.symbol == symbol]


    symbols = sorted(pd_hold_act.symbol.unique())
    pd_price_info_list = []
    time_start = time.time()

    for i_symbol, symbol in zip(range(len(symbols)), symbols):
        pd_hold_act_symbol = pd_hold_act.loc[pd_hold_act.symbol == symbol].sort_values(by='date_earn')
        date_start = pd_hold_act_symbol.date_earn.min()
        date_end = pd_hold_act_symbol.date_earn.max()
        pd_price_ori = stock_price.get_price_range(symbol, date_start, date_end, time_type='date')
        pd_price_ori['adjlow'] = pd_price_ori['low'] * pd_price_ori['close'] / pd_price_ori['adjclose']
        pd_price_ori['time'] = pd_price_ori['time'].str[:10]
        pd_price_ori = pd_price_ori[['time', 'adjclose', 'adjlow']]
        pd_price_list, date_start, date_section = [], '1979-01-01', '1979-01-01'
        status, status_pre = 'sell', 'sell'

        for i in range(len(pd_hold_act_symbol)):
            pd_entry = pd_hold_act_symbol.iloc[i]
            status = pd_entry['act']
            if i == 0:
                date_start = pd_entry['date_earn']
                date_section = date_start
            elif status not in ['sell', 'end']:
                if status_pre != 'sell':
                    date_end = pd_entry['date_earn']
                    pd_price_section = pd_price_ori.loc[(pd_price_ori.time >= date_start) & (pd_price_ori.time < date_end)].copy()
                    if is_update:
                        pd_price_section['date_earn'] = pd_hold_act_symbol.iloc[i - 1]['date_earn']
                    else:
                        pd_price_section['date_earn'] = date_section
                    pd_price_section['act'] = pd_hold_act_symbol.iloc[i - 1]['act']
                    pd_price_list.append(pd_price_section)
                    date_start = pd_entry['date_earn']
                else:
                    date_start = pd_entry['date_earn']
                    date_section = date_start
            elif status in ['sell', 'end']:
                date_end = pd_entry['date_earn']
                pd_price_section = pd_price_ori.loc[(pd_price_ori.time >= date_start) & (pd_price_ori.time <= date_end)].copy()
                if is_update:
                    pd_price_section['date_earn'] = pd_hold_act_symbol.iloc[i - 1]['date_earn']
                else:
                    pd_price_section['date_earn'] = date_section
                pd_price_section['act'] = pd_hold_act_symbol.iloc[i - 1]['act']
                pd_price_section.loc[pd_price_section.index.max(), 'act'] = status
                pd_price_list.append(pd_price_section)
            status_pre = status

        pd_price = pd.concat(pd_price_list)
        pd_price = pd_price.rename(columns={'time': 'date', 'adjclose': 'price_close', 'adjlow': 'price_low'})
        pd_hold_act_symbol = pd_hold_act_symbol.rename(columns={'date': 'date_fisical'})
        keys = [i for i in pd_hold_act_symbol.keys() if i not in ['act', 'price_close', 'price_low']]

        pd_price_info_symbol = pd_price.merge(pd_hold_act_symbol[keys].dropna(), on='date_earn', how='inner')
        pd_price_info_list.append(pd_price_info_symbol)
        time_span = round(time.time() - time_start, 1)
        print(f'\rGetting fine price info - time {time_span} s - progress {i_symbol + 1}/{len(symbols)}', end='')

    print()
    pd_price_info_ori = pd.concat(pd_price_info_list)

    date_earn_list_1 = list(pd_price_info_ori.date_earn.unique())
    date_earn_list_2 = list(pd_price_info_ori.loc[pd_price_info_ori.act.isin(['sell', 'end'])].date.unique())
    date_earn_list = date_earn_list_2 + date_earn_list_1
    pd_price_info = pd_price_info_ori.loc[(pd_price_info_ori.date.isin(date_earn_list))].copy()
    pd_price_info = pd_price_info.sort_values(by=['date', 'symbol'])
    pd_price_info = pd_price_info.rename(columns={'date': 'time'})
    pd_price_info['growth'] = pd_price_info['TangibleBookValue-diff-0'] / pd_price_info['TangibleBookValue-0']



    def cal_trading_weight(pd_hold_act, min_growth=0.3, max_growth=20, growth_slope=35, max_pb_1=35, max_pb_2=50):
        growth = pd_hold_act['TangibleBookValue-diff-0'] / pd_hold_act['TangibleBookValue-0']
        pb = pd_hold_act['PB']
        weight_cal = ((1 + (np.minimum(np.maximum(growth, min_growth), max_growth) - min_growth)
                       / (max_growth - min_growth) * growth_slope) *
                      (1 + (np.maximum(np.minimum(pb, max_pb_2), max_pb_1) - max_pb_1) / (max_pb_2 - max_pb_1)))
        return weight_cal


    pd_price_info['weight'] = cal_trading_weight(pd_price_info, min_growth=0.3, max_growth=10, growth_slope=20, max_pb_1=15, max_pb_2=35)
    pd_price_info.loc[pd_price_info.act == 'sell', 'weight'] = 0

    keys_front = ['growth', 'PB', 'weight']
    pd_price_info = pd_price_info[keys_front + [i for i in pd_price_info.keys() if i not in keys_front]]

    pd_stock_bs = pd_price_info.copy()
    symbols = sorted(pd_price_info.symbol.unique())
    dict_pd_stock_bs = {}
    for symbol in symbols:
        dict_pd_stock_bs[symbol] = pd_stock_bs.loc[pd_stock_bs.symbol == symbol]

    pd_reward = bu_se_2_profit_fine(dict_pd_stock_bs, pd_stock_bs, n_stock_max=5)


    fig, _ax = plt.subplots(1, 1)
    ax = fig.axes
    ax[0].plot(pd_reward.date, pd_reward.asset, '.')
    ax[0].set_yscale('log')
    n_year = (pd_reward.date.iloc[-1] - pd_reward.date.iloc[0]).days / 365
    gain = (pd_reward.asset.iloc[-1]/pd_reward.asset.iloc[0])
    print(gain ** (1 / n_year))

    symbols = list(pd_stock_bs.symbol.unique())
    pd_temp_list = []
    for symbol in symbols:
        pd_temp = pd_stock_bs.loc[pd_stock_bs.symbol == symbol]
        pd_temp_diff = pd_temp.weight.diff()
        pd_temp_list.append(pd_temp.loc[pd_temp_diff.loc[(pd_temp_diff.abs()>0)|(pd_temp_diff.isna())].index])
    pd_stock_temp_final = pd.concat(pd_temp_list).sort_values(by=['time', 'symbol'])

    usd_total = 90000
    _dict_recommend_val, _dict_recommend_num = pd_reward.iloc[-2].hold_val, pd_reward.iloc[-2].hold_num
    _keys = list(_dict_recommend_val.keys())
    dict_recommend = {'symbol': _keys, 'num': [_dict_recommend_num[i] for i in _dict_recommend_num]}
    pd_recommend = pd.DataFrame(dict_recommend)
    pd_price_info_1 = pd_stock_bs.loc[pd_stock_bs.time == pd_stock_bs.time.max()]
    pd_recommend = pd_recommend.merge(pd_price_info_1, on='symbol', how='inner')[['symbol', 'price_close', 'num']].drop_duplicates()
    pd_recommend['usd'] = pd_recommend['price_close'] * pd_recommend['num']
    _rate = usd_total / pd_recommend.usd.sum()
    pd_recommend[['usd', 'num']] = pd_recommend[['usd', 'num']] * _rate
    pd_recommend = pd_recommend.sort_values(by='usd', ascending=False)
    pd_recommend['usd_int'] = 0
    pd_recommend['num_int'] = pd_recommend['num'].round()
    pd_recommend['usd_int'] = pd_recommend['price_close'] * pd_recommend['num']
    pd_recommend = pd_recommend.set_index('symbol')
    pd_recommend_int = pd_recommend[['num_int', 'usd_int']].copy()
    pd_recommend_int['price'] = pd_recommend_int['usd_int'] / pd_recommend_int['num_int']
    pd_recommend_int['usd_int'].sum()



label = 0
if label == 1:
    plt.hist(np.log10(pd_hold_act['TangibleBookValue-diff-0'] / pd_hold_act['TangibleBookValue-0']), bins=50)
    plt.hist(np.log10(pd_hold_act['FreeCashFlow-diff-0'] / pd_hold_act['TangibleBookValue-0']) + 1.5, bins=50)
    plt.hist(np.log10(pd_hold_act['FreeCashFlow-0'] / pd_hold_act['TangibleBookValue-0']) + 1.5, bins=50)
    plt.hist(np.log10(pd_hold_act['GrossProfit-diff-0'] / pd_hold_act['TangibleBookValue-0']) + 1.5, bins=50)
    plt.hist(np.log10(pd_hold_act['OperatingCashFlow-diff-0'] / pd_hold_act['TangibleBookValue-0']) + 1.5, bins=50)
    plt.hist(np.log10(pd_hold_act['PB']), bins=50)

