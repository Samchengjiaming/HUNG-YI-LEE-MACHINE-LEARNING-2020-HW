
#coding=utf-8
'''
@Time: 2021/5/29 15:08  
@Author: 多来B.梦
@File: a.py
Software: PyCharm
target: 程序目标
'''

import numpy as np

a=np.array([[1],[2],[3],[4]])
b=np.array([[1,2,3,4]])
a=a.reshape(1,4)
print(a-b)