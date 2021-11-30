__author__ = 'Yunsong Xie'
__email__ = 'xiefinance00@gmail.com'
__company__ = 'Xie Finance LLC'


import re, os, sys, datetime, pickle, sqlite3
import numpy as np
import pandas as pd
import time
import glob
import lib as common_func

pd.set_option('display.max_column', 60)
pd.set_option('display.max_colwidth', 1200)
pd.set_option('display.width', 12000)

DIR = common_func.misc.get_main_dir()
stock_price = common_func.StockPrice()

dir_fr = f'{DIR}\\static\\Financial_reports\\Wharton\\'
path_csv = max(glob.glob(f'{dir_fr}/wharton_FR_*.csv'))

pd_exclude_keywords = pd.read_csv(f'{DIR}/static/csv/Wharton/exclude_keywords.csv')

path_fr_db = f'{dir_fr}/fr_wharton.db'


try:
    _ = pd.read_sql("select * from col_name limit 10", con)
except:
    con = sqlite3.connect(path_fr_db)


def get_column_info(path_columns):
    """
    Output the column information. No input needed.
    Returns:
        (pandas.dataframe): column info pandas dataframe, contains columns: Variable Name, Type, Description
        (dict): all columns - column vs. description e.g. {'txtq': 'Income Taxes - Total'}
        (dict): all columns - column vs. type e.g. {'txtq': 'double'}
        (set): the columns with date info - column vs. type e.g. {'apdedateq', 'dldte'}
    """
    pd_columns = pd.read_csv(path_columns)
    pd_columns.loc[pd_columns['Variable Name'] == 'TIC', 'Variable Name'] = 'SYMBOL'
    pd_columns = pd_columns.sort_values(by='Variable Name')

    desc_list = list(pd_columns['Description'])
    desc_list = [' ('.join(i.split(' -- ')[-1].split(' (')[:-1]) for i in desc_list]
    pd_columns['desc'] = desc_list
    dict_columns = pd_columns.set_index('Variable Name')['desc'].to_dict()
    dict_columns = {i.lower(): dict_columns[i] for i in dict_columns}
    dict_types = pd_columns.set_index('Variable Name')['Type'].to_dict()
    set_types_date = {i.lower() for i in dict_types if dict_types[i] == 'date'}
    dict_types = {i.lower(): dict_types[i] if dict_types[i] != 'date' else 'str' for i in dict_types}

    print('Completed reading column info')

    return pd_columns, dict_columns, dict_types, set_types_date


def read_report_data(path_csv, path_columns):
    """
    Read Wharton report in csv format, and then output the simplifed columns.
    Simplifed column use exclude_keywords.csv file.

    Args:
        path_csv (str): path of the Wharton report csv file.

    Returns:
        (pandas.dataframe): Simplifed Wharton financial report
    """
    pd_columns, dict_columns, dict_types, set_types_date = get_column_info(path_columns)
    print('Start reading Wharton financial report csv file')
    pd_wharton = pd.read_csv(path_csv).rename(columns={'tic': 'symbol'})
    print('Completed reading Wharton financial report csv file')

    columns = list(pd_wharton.columns)
    columns_in = [i for i in columns if i in dict_columns]
    pd_wharton_fr = pd_wharton[columns_in]
    pd_wharton_fr = pd_wharton_fr.astype(dict_types)
    for key in set_types_date:
        pd_wharton_fr[key] = pd_wharton_fr[key].str[:8]

    columns_exclude_keywords = list(pd_exclude_keywords.keyword)
    cols_select = list(dict_columns)
    for keyword in columns_exclude_keywords:
        cols_select = [i for i in cols_select if keyword not in dict_columns[i].lower()]
    code_exclude_keywords = [i.split('(code)@')[1] for i in columns_exclude_keywords if ('(code)@' in i)]
    cols_select = [i for i in cols_select if i not in code_exclude_keywords]

    pd_wharton_fr = pd_wharton_fr[cols_select]

    # Report date of quarterly earnings can't be empty
    pd_wharton_fr = pd_wharton_fr.loc[pd_wharton_fr['rdq'] != 'nan'].copy()

    pd_columns_select = pd_columns.loc[pd_columns['Variable Name'].str.lower().isin(list(pd_wharton_fr.columns))]
    dict_types_select = pd_columns_select.set_index('Variable Name')['Type'].to_dict()
    dict_types_select_str = {i.lower() for i in dict_types_select if dict_types_select[i] in ['string', 'date']}
    for i in dict_types_select_str:
        pd_wharton_fr.loc[pd_wharton_fr[i] == 'nan', i] = ''
        pd_wharton_fr.loc[pd_wharton_fr[i].isna(), i] = ''

    return pd_wharton_fr


def finalize_report(pd_wharton_fr, write_out=False):
    """
    Finalize the Wharton financial report. This function does the two things
    1. Re-arange the column sequence
    2. change the date format from 20000101 to 2000-01-01.
    Args:
        pd_wharton_fr (pandas.dataframe): output from simplify_report function, it has excluded all unneeded columns
        write_out (Bool): Whether to write out to pickle

    Returns:
        (pandas.dataframe):
    """
    columns = sorted(pd_wharton_fr.columns)
    col_head = ['symbol', 'rdq', 'datafqtr', 'pdateq', 'datacqtr', 'fdateq']
    columns_final = col_head + [i for i in columns if i not in col_head]
    pd_wharton_fr = pd_wharton_fr[columns_final]
    cols_date_num = ['rdq', 'pdateq', 'fdateq']
    for col in cols_date_num:
        inds = pd_wharton_fr[col] != ''
        pd_wharton_fr.loc[inds, col] = (pd_wharton_fr.loc[inds, col].str[:4] + '-' +
                                        pd_wharton_fr.loc[inds, col].str[4:6] + '-' +
                                        pd_wharton_fr.loc[inds, col].str[6:])

    date_now = str(datetime.datetime.now())[:10].replace('-', '')
    if write_out:
        pd_wharton_fr.to_pickle(f'{dir_fr}/wharton_FR_{date_now}.pkl').to_pickle(f'{dir_fr}/wharton_FR_{date_now}.pkl')
    print('Completed finalizing Wharton financial report data')
    return pd_wharton_fr


def read_column_data(path_columns):
    """
    Read column info table for the Wharton financial report

    Args:
        path_columns (str): path to the column info.

    Returns:
        (pandas.dataframe): column info from Wharton financial report
    """

    pd_columns = pd.read_csv(path_columns)
    pd_columns.loc[pd_columns['Variable Name'] == 'TIC', 'Variable Name'] = 'SYMBOL'
    pd_columns = pd_columns.sort_values(by='Variable Name').rename(columns={'Variable Name': 'col_name'})
    pd_columns['col_name'] = pd_columns['col_name'].str.lower()

    desc_list = list(pd_columns['Description'])
    desc_list = [' ('.join(i.split(' -- ')[-1].split(' (')[:-1]) for i in desc_list]
    pd_columns['desc'] = desc_list

    return pd_columns


def upload_report_data(path_fr_db, pd_wharton_fr, pd_columns):
    """
    Upload Wharton financial report to db file, programe will automatically upload the data does not
    exist in the db file.
    Args:
        path_fr_db (str): path of the db file
        pd_wharton_fr (pandas.dataframe): uploading data
    """
    con = sqlite3.connect(path_fr_db)

    # type info for each column in the report data frame
    pd_type = check_db_file(path_fr_db, pd_wharton_fr, pd_columns)

    merge_cols = ['symbol', 'rdq', 'datafqtr']
    command_query = f"""select {', '.join(merge_cols)} from report"""
    pd_exist = pd.read_sql(command_query, con)
    temp_col = 'temp'
    pd_exist[temp_col] = 'e'
    pd_fr_upload = pd_wharton_fr.merge(pd_exist, on=merge_cols, how='left').copy()
    pd_fr_upload = pd_fr_upload.loc[pd_fr_upload[temp_col].isna()][[i for i in pd_fr_upload.columns if i != temp_col]]

    columns = list(pd_fr_upload.columns)
    dict_type = pd_type.set_index('col_name')['dtype'].to_dict()
    time_start = time.time()
    for i_col, col in enumerate(columns):
        if dict_type[col] == 'string':
            pd_fr_upload[col] = '"' + pd_fr_upload[col] + '"'
        else:
            pd_fr_upload[col] = pd_fr_upload[col].astype(str)
            pd_fr_upload.loc[pd_fr_upload[col] == 'nan', col] = 'NULL'
        time_span = int(round(time.time() - time_start))
        print(f'\rTime: {time_span} s - Processing original data frame - Progress {i_col + 1}/{len(columns)}', end='')
    if len(pd_fr_upload) > 0:
        print()
    batch_size = 1000
    n_batch = int(np.ceil(len(pd_fr_upload)/batch_size))
    time_start = time.time()
    for i in range(n_batch):
        pd_batch = pd_fr_upload.iloc[i * batch_size: (i + 1) * batch_size]
        command = f"""INSERT INTO report ({', '.join(columns)}) values \n"""
        data_batch = pd_batch.values
        for j in range(len(data_batch)):
            command += f"({', '.join(data_batch[j])}), \n"

        command = command[:-3]
        con.execute(command)
        con.commit()
        time_span = int(round(time.time() - time_start))
        print(f'\rTime: {time_span} s - Uploading data to db file - Progress {i + 1}/{n_batch}', end='')
    if n_batch > 0:
        print()
    con.close()


def check_db_file(path_fr_db, pd_wharton_fr, pd_columns):
    """
    Establish db file both data table and index table
    Args:
        path_fr_db (str): path to the db file
        pd_type (pandas.dataframe): type and desc info of each column
    """
    pd_type = pd_wharton_fr.dtypes.reset_index().rename(columns={'index': 'col_name', 0: 'dtype'})
    pd_type['dtype'] = pd_type['dtype'].astype(str)
    pd_type.loc[pd_type.dtype == 'object', 'dtype'] = 'string'
    pd_type = pd_type.merge(pd_columns[['col_name', 'desc']], on='col_name', how='inner')

    if not os.path.isfile(path_fr_db):
        con = sqlite3.connect(path_fr_db)
        command = """CREATE TABLE "report" ("""
        for i in range(len(pd_type)):
            col, dtype = pd_type.iloc[i][['col_name', 'dtype']]
            dtype = 'TEXT' if dtype == 'string' else 'NUMERIC'
            command += f''' "{col}"	{dtype}, \n'''
        command = command + 'PRIMARY KEY("symbol", "rdq", "datafqtr") )'
        con.execute(command)

        command = """CREATE TABLE "col_name" ("seq" INTEGER, "col_name" TEXT, "desc" TEXT)"""
        con.execute(command)
        con.commit()

        command = """INSERT INTO col_name (seq, col_name, desc) values """
        for i in range(len(pd_type)):
            col_name, desc = pd_type.iloc[i][['col_name', 'desc']]
            command += f"""({i + 1}, "{col_name}", "{desc}"), \n"""
        command = command[:-3]
        con.execute(command)
        con.commit()

        command = """create index 'report_index' on 'report' ('symbol', 'datafqtr')"""
        con.execute(command)
        con.commit()

        command = """CREATE TABLE "log_transactions" ("time" TEXT, "table_name" TEXT, "file_name" TEXT);"""
        con.execute(command)
        con.commit()

        con.close()
    return pd_type


if __name__ == '__main__':
    # Read Wharton financial report data
    path_csv = max(glob.glob(f'{dir_fr}/wharton_FR_*.csv'))
    print(f'Uploading file: {os.path.basename(path_csv)}')
    path_columns = glob.glob(f'{dir_fr}/wharton_columns.csv')[0]
    pd_wharton_fr = read_report_data(path_csv, path_columns)
    pd_wharton_fr = finalize_report(pd_wharton_fr, write_out=False)

    # Read Wharton financial report data column info table

    pd_columns = read_column_data(path_columns)

    # Check condition of the db
    pd_type = check_db_file(path_fr_db, pd_wharton_fr, pd_columns)

    # Start uploading
    print('Completed checking db file')
    upload_report_data(path_fr_db, pd_wharton_fr, pd_columns)
    print('Completed uploading data to db file')

    stock_price.upload_transaction('report', path_csv.replace('\\', '/'))