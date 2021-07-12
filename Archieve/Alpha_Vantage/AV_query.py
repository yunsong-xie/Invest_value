# Author: Yunsong Xie


import requests, os, sqlite3
import datetime, time
import pandas as pd
import numpy as np
import robin_stocks as rs

DIR = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
rs.login(username='xieyunsong@gmail.com', password='Mitimt#325',
         expiresIn=864000000, by_sms=True, backup_code='313749 217812')

pd.set_option('display.max_column', 10)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)



# &symbol=IBM&apikey=
dict_query_json = {'balance_sheet': 'https://www.alphavantage.co/query?function=BALANCE_SHEET',
                   'income_statement': 'https://www.alphavantage.co/query?function=INCOME_STATEMENT',
                   'cash_flow': 'https://www.alphavantage.co/query?function=INCOME_STATEMENT',
                   'overview': 'https://www.alphavantage.co/query?function=OVERVIEW',
                   'price_month': 'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED'}

#https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&symbol=IBM&horizon=12month&apikey=demo
dict_query_csv = {'earning_calendar_3month': 'https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month',
                  'earning_calendar_12month': 'https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=12month', }

API_KEY = 'WWNEDKODZSJ6O6NW'
API_KEY = 'UVB0DO7YW47OR9E4'
BASE_URL = 'https://www.alphavantage.co/query?'
params = {'function': 'cash_flow', 'symbol': 'IBM', 'apikey': API_KEY}
PATH_DB = f'{DIR}/assets/stock_financial.db'
# response = requests.get(base_url, params=params)
# data = response.json()


class financial_data:
    def __init__(self):
        pd_fm = self.get_fundamental()
        pd_fm = pd_fm.sort_values(by='market_cap', ascending=False)
        self.pd_fm = pd_fm
        self.pd_fm_us = pd_fm.loc[pd_fm.country == 'US']
        self.pd_ciq = self.get_ciq(self.pd_fm)
        self.pd_ciq = self.pd_ciq.sort_values(by='market_cap', ascending=False)
        self.pd_ciq_us = self.get_ciq(self.pd_fm_us)
        self.conn = sqlite3.connect(PATH_DB)

    @staticmethod
    def get_fundamental():
        """
        This function queries the fundamental of all active stocks in NYSE and NASDAQ
        Returns:
            (Pandas Dataframe): stock fundamental data
        """
        csv_list = [i for i in os.listdir(f'{DIR}/assets/csv') if '.csv' == i[-4:] and 'fundamentals' in i]
        label_reload = True
        date_now = datetime.datetime.now()
        date_now_str = str(date_now).replace('-', '')[:8]
        if csv_list:
            # Determine whether to reload the listing stock database
            date_last_load_str = max(csv_list).split('_')[-1][:8]
            date_last_load = datetime.datetime.strptime(date_last_load_str, '%Y%m%d')
            date_num = int(np.ceil((date_now - date_last_load).total_seconds()/(24*3600)))
            if date_num < 28:
                label_reload = False
                pd_fm = pd.read_csv(f'{DIR}/assets/csv/{max(csv_list)}')

        if label_reload:
            # Get active stock listing info via alpha vantage
            pd_listing = pd.read_csv('https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=demo')
            pd_listing.to_csv(f'{DIR}/assets/csv/listing_status_{date_now_str}.csv', index=False)

            pd_listing = pd_listing.loc[(pd_listing.exchange.isin(['NYSE', 'NASDAQ'])) & (pd_listing.assetType == 'Stock')]
            pd_listing_1 = pd_listing.loc[~pd_listing.symbol.str.contains('-')]
            symbols = list(pd_listing_1.symbol.unique())

            def get_fundamentals(_symbols):
                """
                Obtain stock fundamental data columns include:
                symbol, market_cap, headquarters_city, headquarters_state, sector, industry, country

                Args:
                    _symbols(list): symbols to be studied

                Returns:
                    (Pandas Dataframe): stock fundamental data
                """
                if type(_symbols) is str:
                    _symbols = [_symbols]
                elif len(_symbols) > 100:
                    raise ValueError("Length of symbol list can not exceed 100")
                fundamentals = rs.get_fundamentals(_symbols)
                _pd_fm = pd.DataFrame(fundamentals)
                _pd_fm = _pd_fm[['symbol', 'market_cap', 'headquarters_city', 'headquarters_state', 'sector', 'industry']]
                _pd_fm = _pd_fm.loc[~_pd_fm.market_cap.isnull()]
                _symbols_new = list(_pd_fm.symbol)
                pd_stock_country = pd.DataFrame(rs.get_instruments_by_symbols(_symbols_new))[['symbol', 'country']]
                _pd_fm = pd_stock_country.merge(_pd_fm, on='symbol', how='inner')
                return _pd_fm
            n_entry, pd_fm_list = 100, []
            time_start = time.time()
            n_total = int(np.ceil(len(symbols)/n_entry))
            for i in range(n_total):
                # Robinhood can only process 100 stocks once, need to split the data query into multiple sections
                pd_fm_entry = get_fundamentals(symbols[(i*n_entry):((i+1)*n_entry)])
                pd_fm_list.append(pd_fm_entry)
                time_span = round(time.time() - time_start)
                print(f'\rSequence {i+1}/{n_total} - {time_span} s', end='')

            pd_fm = pd.concat(pd_fm_list)
            pd_fm['market_cap'] = pd_fm['market_cap'].astype(float)/10**9
            pd_fm = pd_fm.sort_values(by='market_cap', ascending=False)
            pd_fm.index = np.arange(len(pd_fm))
            pd_fm = pd_listing_1[['symbol', 'exchange']].merge(pd_fm, on='symbol', how='inner')
            pd_fm.to_csv(f'{DIR}/assets/csv/fundamentals_{date_now_str}.csv', index=False)

        exclude_list = ['GOOG', ]
        pd_fm = pd_fm.loc[~pd_fm.symbol.isin(exclude_list)]
        return pd_fm

    @staticmethod
    def get_ciq(pd_fm):
        """
        Use the exchange ticker info obtained from capital IQ to compile the final pd_ciq,
        the dataframe contains the followed columns:

        ['symbol', 'exchange_ticker', 'exchange', 'country', 'market_cap',
        'headquarters_city', 'headquarters_state', 'sector', 'industry']

        Args:
            pd_fm (Pandas dataframe): fundamental dataframe from alpha vantage and robinhood.

        Returns:
            (Pandas dataframe): final pd_ciq
        """
        pd_temp = pd.read_csv(f'{os.path.dirname(DIR)}/capitalIQ/static/csv/stock_list/Capital_IQ_exchange.csv', sep='\t')
        pd_temp['info'] = pd_temp['info'].str.replace('*', '')
        symbol_list = []
        for i in range(len(pd_temp)):
            info = pd_temp.iloc[i]['info']
            symbol_info = [i for i in info.split(';') if (('NYSE' in i) | ('Nasdaq' in i)) & ('.' not in i)]
            if len(symbol_info) == 1:
                symbol = symbol_info[0]
            elif len(symbol_info) == 0:
                symbol = info[0]
            else:
                symbol = symbol_info[0]
            symbol_list.append(symbol.split(':')[-1])
        pd_ciq = pd.DataFrame({'symbol': symbol_list, 'exchange_ticker': list(pd_temp.exchange_ticker)})
        pd_ciq = pd_ciq.merge(pd_fm, on='symbol', how='inner')
        return pd_ciq

    @staticmethod
    def _create_db():
        conn = sqlite3.connect(PATH_DB)
        table_list = ['cashflow', 'income', 'balance']
        span_list = ['annual', 'quarter']
        for table in table_list:
            for span in span_list:
                command = f"""
                CREATE TABLE "{table}_{span}" (
                "symbol"	TEXT NOT NULL,
                "item"	TEXT NOT NULL,
                "value"	NUMERIC,
                "time"	TEXT NOT NULL,
                PRIMARY KEY("symbol")
                )
                """
                conn.cursor().execute(command)

        command = f"""
                CREATE TABLE "currency" (
                "symbol"	TEXT NOT NULL,
                "currency"	TEXT NOT NULL,
                "time"	TEXT NOT NULL,
                PRIMARY KEY("symbol")
                )
                """
        conn.cursor().execute(command)
        conn.commit()
        conn.close()

    def _check_table(self, table):
        try:
            pd.read_sql(f'select * from {table} limit 10', self.conn)
        except Exception as _:
            pd_financial = self.get_financial(table, 'AAPL')

            command = f"""CREATE TABLE "{table}" (\n"""
            pd_type = pd_financial.dtypes
            for key in list(pd_financial.keys()):
                if str(pd_type[key]) == 'object':
                    col_type = 'TEXT'
                else:
                    col_type = 'NUMERIC'

                command += f""" "{key}" {col_type} NOT NULL, \n"""
            command = command[:-3] + ')'
            self.conn.cursor().execute(command)
            self.conn.commit()

    @staticmethod
    def get_AV_data(table, symbol):
        params = {'function': table, 'symbol': symbol, 'apikey': API_KEY}
        response = requests.get(BASE_URL, params=params)
        AV_data = response.json()
        return AV_data

    @staticmethod
    def get_financial(AV_data, table, symbol):

        pd_financial_a = pd.DataFrame(AV_data['annualReports'])
        pd_financial_a = pd_financial_a.replace('None', '0')
        pd_financial_a['fiscalDateEnding'] = (pd_financial_a['fiscalDateEnding'].str[:4].astype(int))

        pd_financial_q = pd.DataFrame(AV_data['quarterlyReports'])
        pd_financial_q = pd_financial_q.replace('None', '0')
        pd_financial_q['fiscalDateEnding'] = (pd_financial_q['fiscalDateEnding'].str[:4].astype(int)) + \
                                            (pd_financial_q['fiscalDateEnding'].str[5:7]).astype(int)/12
        pd_financial_a['timeType'] = 'annual'
        pd_financial_q['timeType'] = 'quarter'

        pd_financial = pd.concat([pd_financial_a, pd_financial_q])
        pd_financial['symbol'] = symbol

        header_cols = ['symbol', 'timeType']
        for key in list(pd_financial.keys()):
            if key not in ['reportedCurrency'] + header_cols:
                pd_financial[key] = pd_financial[key].astype(float)
                pd_financial[key] = pd_financial[key].astype(float)

        pd_financial = pd_financial[header_cols + [i for i in list(pd_financial.keys()) if i not in header_cols]]
        pd_financial = pd_financial.rename(columns={'fiscalDateEnding': 'time'})

        return pd_financial

    def update_table(self, table, pd_financial):
        merge_cols = ['time', 'timeType', 'symbol']
        if len(pd_financial) > 0:

            self._check_table(table)

            symbols = list(pd_financial.symbol.unique())

            command_query = f"""select time, timeType, symbol from {table} where symbol in ("""
            for symbol in symbols:
                command_query += f'"{symbol}", '
            command_query = command_query[:-2] + ')'
            pd_exist = pd.read_sql(command_query, self.conn)
            pd_exist['temp'] = 1

            pd_upload = pd_financial.merge(pd_exist, on=merge_cols, how='left')
            pd_upload = pd_upload.loc[pd_upload.temp.isna(), list(pd_financial.keys())]

            if len(pd_upload) > 0:
                keys = list(pd_upload.keys())
                command_col_str = ', '.join(keys)
                command_insert = f"""insert into {table} ({command_col_str}) values """
                for i in range(len(pd_upload)):
                    pd_entry = pd_upload.iloc[i]
                    value_list = [f"{i}" if type(i) is np.float64 else f'"{i}"'  for i in list(pd_entry[keys])]
                    command_row_str = ', '.join(value_list)
                    command_insert += f""" ({command_row_str}),\n"""
                command_insert = command_insert[:-2]

                self.conn.cursor().execute(command_insert)
                self.conn.commit()

    def main(self):
        pd_fm = self.get_fundamental()
        pd_fm_us = pd_fm.loc[pd_fm.country == 'US']

        symbols = list(pd_fm_us.symbol)
        command = "select distinct symbol from cash_flow"
        symbols_exist = list(pd.read_sql(command, self.conn).symbol)

        symbols_update = [i for i in symbols if i not in symbols_exist]

        tables = ['balance_sheet', 'income_statement', 'cash_flow']
        dict_finance = {}
        dict_AV_data = {}
        time_start = time.time()
        count_query, n_symbol, n_table = 0, len(symbols_update), 3
        for ind_symbol, symbol in zip(range(n_symbol), symbols_update):
            for ind_table, table in zip(range(n_table), tables):
                dict_AV_data[table] = self.get_AV_data(table, symbol)
                _pd_finance = self.get_financial(dict_AV_data[table], table, symbol)
                dict_finance[table] = _pd_finance
                self.update_table(table, dict_finance[table])
                time_span = round(time.time() - time_start)
                count_query += 1
                print(f'\r{time_span} s - {ind_symbol+1}/{n_symbol} {symbol} - {ind_table+1}/{n_table} {table} - '
                      f'query count: {count_query}', end='')
                time.sleep(12)

