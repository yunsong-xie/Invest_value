__author__ = 'Yunsong Xie'
__email__ = 'xiefinance00@gmail.com'
__company__ = 'Xie Finance LLC'

import os, pyodbc, sqlite3

import pandas as pd


def get_main_dir():
    folders_exist = ['lib', 'cache', 'scripts', 'static', '.gitignore']
    dir_cur = os.path.dirname(os.path.abspath(__file__))
    while len(dir_cur) >= 3:
        dir_cur = os.path.dirname(dir_cur)
        folders = os.listdir(dir_cur)
        labels = [(i in folders) for i in folders_exist]
        if all(labels):
            return dir_cur
    return None


DIR_MAIN = get_main_dir()
PATH_DB = f'{DIR_MAIN}/static/Financial_reports/Wharton/fr_wharton.db'

def get_login_info():
    """
    Get the login info for varies of websites
    Returns:
        (dict): [([website], [type]): [info]]
    """
    dir_home = os.path.expanduser('~')
    path_login_info = f'{dir_home}/login_info.csv'
    pd_login = pd.read_csv(path_login_info)
    pd_login = pd_login.set_index(['website', 'item'])
    dict_login = pd_login.to_dict()['value']
    return dict_login


def get_sql_con(conn=None, db_type='local'):
    """
    Get the azure sql connection
    Returns:
        (pyodbc.Connection): connection handle of the azure sql database
    """
    if conn is None:
        reload = True
    else:
        try:
            _ = pd.read_sql("SELECT * FROM col_name limit 1", conn)
            return conn
        except ConnectionError as e:
            reload = True

    if reload:
        if db_type.lower() == 'cloud':
            dict_login = get_login_info()

            server = dict_login[('azure_sql', 'server')]
            database = dict_login[('azure_sql', 'database')]
            username = dict_login[('azure_sql', 'username')]
            password = dict_login[('azure_sql', 'password')]
            driver = dict_login[('azure_sql', 'driver')]

            con_info = f"DRIVER={driver};SERVER=tcp:{server};PORT=1433;DATABASE={database};UID={username};PWD={password}"
            conn = pyodbc.connect(con_info)
        elif db_type.lower() == 'local':
            conn = sqlite3.connect(PATH_DB)
        else:
            raise ValueError('Invalid db_type, it should be either cloud or local. ')

    return conn
