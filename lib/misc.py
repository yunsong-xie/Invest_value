__author__ = 'Yunsong Xie'
__email__ = 'xiefinance00@gmail.com'
__company__ = 'Xie Finance LLC'

import os

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