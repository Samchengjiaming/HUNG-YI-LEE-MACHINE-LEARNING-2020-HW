# coding=utf-8
'''
@Time: 2021/5/30 19:20  
@Author: 多来B.梦
@File: linear model_2.py
Software: PyCharm
target: 程序目标
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


w1=np.zeros([1,9*18+1])
w2=np.zeros([1,9*18])
x_1=np.concatenate((x,np.ones([1,5751])),axis=0)
x_2=x**2
epoch_nums=1000
learning_rate=100
esp=0.0001
adagrad_w1=np.zeros([9*18+1,1])
adagrad_w2=np.zeros([9*18,1])
loss_list=list()
for epoch in range(epoch_nums):
    loss=np.sqrt(np.sum(np.power((y_true-(np.dot(w1,x_1)+np.dot(w2,x_2))),2))/5751)
    if epoch%10==0:
        print('loss:',loss)
        loss_list.append(loss)
    gradient_w1_b=-2*np.dot(x_1,(y_true-(np.dot(w1,x_1)+np.dot(w2,x_2))).transpose())
    gradient_w2 =-2*np.dot(x_2,(y_true-(np.dot(w1,x_1)+np.dot(w2,x_2))).transpose())
    adagrad_w1+=gradient_w1_b**2
    adagrad_w2+=gradient_w2**2
    w1=w1-(learning_rate*gradient_w1_b/np.sqrt(adagrad_w1+esp)).reshape(1,163)
    w2=w2-(learning_rate*gradient_w2/np.sqrt(adagrad_w2+esp)).reshape(1,162)

x_index=[i for i in range(len(loss_list))]
plt.plot(x_index, loss_list, color='red', linewidth=2.0, linestyle='-')
plt.title('loss value')
plt.ylabel('loss')
plt.xlabel('index of epoch')
plt.show()

