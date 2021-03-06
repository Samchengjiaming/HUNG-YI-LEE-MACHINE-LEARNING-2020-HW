# coding=utf-8
'''
@Time: 2021/5/27 2:57  
@Author: samchengjiaming
@File: linear model_1.py
Software: PyCharm
target: hw1
'''

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
'''
Due to the problem of the data character encoding set, we changed all Chinese in the data to English, 
but this did not affect the results of our model training.
'''

train_data_set = pd.read_csv('../data/train.csv')
'''
The following code block constructs the input feature X and y^.
'''
all_X_df = pd.DataFrame()
x_batch_list = list()
y_batch_list = list()
for day in range(int(train_data_set.shape[0] / 18)):
    one_day_data = train_data_set.iloc[day * 18:day * 18 + 18, 2:]
    y = one_day_data.iloc[9, :].tolist()
    y_batch_list+=y
    one_day_data.columns = [i for i in range(day * 24, (day + 1) * 24)]
    one_day_data.index = [i for i in range(0, 18)]
    all_X_df = pd.concat([all_X_df, one_day_data], axis=1)
for hour in range(0, all_X_df.shape[1] - 9):
    nine_hours_data = all_X_df.iloc[:, [i for i in range(hour, hour + 9)]]
    x_batch = nine_hours_data.values.reshape((18 * 9, 1))
    x_batch_list.append(x_batch)
y_batch_list = y_batch_list[9:]
y_true=np.array([y_batch_list])
x=np.empty([162,1],dtype=float)
for x_batch in x_batch_list:
    x=np.concatenate((x,x_batch),axis=1)
x=x[:,1:]

'''
Model 1:Linear equation
'''
#将常数项b当做参数w，只不过常熟b对应的x为1
w=np.zeros([1,9*18+1])
x=np.concatenate((x,np.ones([1,5751])),axis=0)
epoch_nums=1000
learning_rate=100
esp=0.0001
adagrad=np.zeros([9*18+1,1])
loss_list=list()
for epoch in range(epoch_nums):
    loss=np.sqrt(np.sum(np.power((y_true-np.dot(w,x)),2))/5751)
    if epoch%10==0:
        print('loss:',loss)
        loss_list.append(loss)
    gradient=-2*np.dot(x,(y_true-np.dot(w,x)).transpose())
    adagrad+=gradient**2
    w=w-(learning_rate*gradient/np.sqrt(adagrad+esp)).reshape(1,163)

x_index=[i for i in range(len(loss_list))]
plt.plot(x_index, loss_list, color='red', linewidth=2.0, linestyle='-')
plt.title('loss value')
plt.ylabel('loss')
plt.xlabel('index of epoch')
plt.show()

