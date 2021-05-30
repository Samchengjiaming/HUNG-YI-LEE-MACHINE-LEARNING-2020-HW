# coding=utf-8
'''
@Time: 2021/5/27 2:57  
@Author: samchengjiaming
@File: linear model.py
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
test_data_set = pd.read_csv('../data/test.csv')

x_test = test_data_set.iloc[:18, 2:]
print(x_test)
'''
The following code block constructs the input feature X and y^.
'''
all_X_df = pd.DataFrame()
x_batch_list = list()
y_batch_list = list()
for day in range(int(train_data_set.shape[0] / 18)):
    one_day_data = train_data_set.iloc[day * 18:day * 18 + 18, 2:]
    y = one_day_data.iloc[9, :].tolist()
    y_batch_list.extend(y)
    one_day_data.columns = [i for i in range(day * 24, (day + 1) * 24)]
    one_day_data.index = [i for i in range(0, 18)]
    all_X_df = pd.concat([all_X_df, one_day_data], axis=1)
for hour in range(0, all_X_df.shape[1] - 9):
    nine_hours_data = all_X_df.iloc[:, [i for i in range(hour, hour + 9)]]
    x = nine_hours_data.values.reshape((18 * 9, 1))
    x_batch_list.append(x)
y_batch_list = y_batch_list[9:]

'''
Model 1:Linear equation
'''
linear_equation_model_b = np.random.randn(1, 1)
linear_equation_model_W1 = np.random.randn(1, 162)
learning_rate = 0.00001
train_error_list=list()

for i in range(1, 6):
    print(i)
    learning_rate=learning_rate/10
    for (y_batch, x_batch) in zip(y_batch_list, x_batch_list):
        for index in range(0, len(linear_equation_model_W1[0])):
            gradient_descent_Wi = 2 * (
                    y_batch - (linear_equation_model_b + np.dot(linear_equation_model_W1, x_batch)[0][0])) * (
                                      -x_batch[index][0])
            new_Wi = linear_equation_model_W1[0][index] - learning_rate * gradient_descent_Wi
            linear_equation_model_W1[0][index] = new_Wi
            gradient_descent_b = 2 * (
                    y_batch - (linear_equation_model_b + np.dot(linear_equation_model_W1, x_batch))) * (-1)
            linear_equation_model_b = linear_equation_model_b - learning_rate * gradient_descent_b[0][0]
            train_error=(y_batch - (linear_equation_model_b + np.dot(linear_equation_model_W1, x_batch)[0][0]))[0][0]
            train_error_list.append(train_error)

x_index=[i for i in range(len(train_error_list))]
plt.plot(x_index, train_error_list, color='red', linewidth=2.0, linestyle='-')
plt.ylabel('error of train')
plt.xlabel('times of batch')
plt.show()
y_test = linear_equation_model_b + np.dot(linear_equation_model_W1, x_test.values.reshape(162, 1))
print(y_test)
