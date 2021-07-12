# Author: Yunsong Xie

import pyEX
token = 'sk_7922457a94de448984045bbb3d605fd9'
token = 'pk_8af39569b51d4fa58d5cbac8d152ad14'

DIR

handle = pyEX.Client(api_token=token, version='stable', api_limit=5)

# data = handle.stocks.balanceSheet('SQ', period='quarter', last=12, format='json')

# pd_data1 = pd.DataFrame(data)
# pd_data1.to_pickle('D:\PycharmProjects\Investment\Invest_value\IEX\static/BalanceSheet_SQ.pkl')
# pd_data1['date'] = pd.to_datetime(pd_data1['date'], unit='ms')
pd_data2 = pd_data1.copy()
keys_exclude = ['capitalSurplus', 'currency', 'treasuryStock', 'key', 'treasuryStock', 'subkey', 'updated',
                'filingType', 'fiscalDate', 'fiscalYear', 'fiscalQuarter', 'id']
pd_data2 = pd_data2[[i for i in list(pd_data2.keys()) if i not in keys_exclude]]