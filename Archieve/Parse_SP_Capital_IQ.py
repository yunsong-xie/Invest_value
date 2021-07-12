
import re
import sqlite3
import datetime, time
from bs4 import BeautifulSoup as bs
import urllib.request
import csv
import numpy as np
import copy

import requests


def make_soup(url):
    thepage = urllib.request.urlopen(url)
    soupdata = bs(thepage, "html.parser")
    return soupdata

input_keys=['__EVENTTARGET', '__EVENTARGUMENT', '__VIEWSTATE', '__VIEWSTATEGENERATOR', '__EVENTVALIDATION', 'captchaValidated', 'PersistentLogin']
login_data={'username': 'Ru.Xie.21@mba.mccombs.utexas.edu',
            'password': 'Tuesday2010',
            }
headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'}

with requests.Session() as s:
    url = 'https://www.capitaliq.com/ciqdotnet/login-sso.aspx?bmctx=202C7940E5B3C784A1D977EF9E24AB0D&contextType=external&username=string&enablePersistentLogin=true&OverrideRetryLimit=0&contextValue=%2Foam&password=secure_string&challenge_url=https%3A%2F%2Fwww.capitaliq.com%2Fciqdotnet%2Flogin-sso.aspx&request_id=3182847596904274431&authn_try_count=0&locale=en_US&resource_url=https%253A%252F%252Fwww.capitaliq.com%252FCIQDotNet%252Flogin.aspx'

    result_get=s.get(url, headers=headers)
    cookies = dict(result_get.cookies)

    soup_1 = bs(str(result_get.text), "html.parser")
    if 1==1:
        for input_key in input_keys:
            try:
                value=soup_1.find('input', attrs={'name': input_key})['value']
            except:
                value=None
            login_data[input_key]=value

    result_2=s.post(url, data=login_data, headers=headers, verify=False, allow_redirects=True)
    time.sleep(0.25)
    result_3=s.get(url='https://www.capitaliq.com/CIQDotNet/Search/Search.aspx')

soup_2 = bs(str(result_2.text), "html.parser")
soup_3 = bs(str(result_3.text), "html.parser")
soup_1_input_dict=soup_1.find_all('input')
soup_2_input_dict=soup_2.find_all('input')
soup_3_input_dict=soup_3.find_all('input')

s.close()

for input_key in input_keys:
    content1 = soup_1.find('input', attrs={'name': input_key})
    content2 = soup_2.find('input', attrs={'name': input_key})
    content3 = soup_3.find('input', attrs={'name': input_key})
    if content3!=content2:
        print(content1)
        print(content2)
        print(content3)
        print(content1==content2)

with open('result_2.html', 'w') as f:
    f.write(str(result_2.text))
f.close()


with open('result_3.html', 'w') as f:
    f.write(str(result_3.text))
f.close()


if 1==0:
    soup=make_soup('https://www.capitaliq.com/CIQDotNet/Financial/IncomeStatement.aspx?CompanyId=32012')

    with open('workfile.html', 'w') as f:
        f.write(str(soup))

    f.close()