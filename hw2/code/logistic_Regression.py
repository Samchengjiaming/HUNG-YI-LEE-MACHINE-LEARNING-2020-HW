#coding=utf-8
'''
@Time: 2021/6/7 15:07  
@Author: 多来B.梦
@File: logistic_Regression.py
Software: PyCharm
target: 程序目标
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import  OneHotEncoder
from sklearn.compose import ColumnTransformer
'''
data preprocessing
1.unnecessary attributes removed
    1.Remove abnormal data,if an attribute contain '?' or 'Not in  Not in universe',we will remove this attribute
2.positive/negative ratio balanced.
'''

train_data=pd.read_csv('../data/train.csv')
# print(train_data.info())

X=pd.DataFrame()
# unnecessary attributes removed
for column in train_data.columns:
    if not train_data[column].values.__contains__(' Not in universe') \
            and not train_data[column].values.__contains__(" ?") \
            and not train_data[column].values.__contains__(' Not in universe under 1 year old') \
            and not train_data[column].values.__contains__(' Not in universe or children'):
        X=pd.concat([X,train_data[column]],axis=1)

'''
get Y_train
'''
y_true_map={
    "-50000":0,
    ' 50000+.':1
}
X.iloc[:,-1]=X.iloc[:,-1].map(y_true_map)
y_true=X.iloc[:,-1]

'''
Use one-hot to get X_train
'''
#one-hot
#先对数据进行数字编码
categorical_feature=['education', 'wage per hour','marital stat',
                     'race','hispanic origin','sex','full or part time employment stat',
                     'tax filer stat','detailed household and family stat','detailed household summary in household',
                     'citizenship']
for column in categorical_feature:
    categorical_feature_set=set(X[column])
    categorical_feature_num_dict={}
    for num,feature in enumerate(categorical_feature_set):
        categorical_feature_num_dict[feature]=num
    X[column]=X[column].map(categorical_feature_num_dict)

#对上述进行数字编码的数据进行one-hot编码
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(train_data[categorical_feature].values)

matrix = enc.transform(train_data[categorical_feature].values).toarray()
feature_labels = np.array(enc.categories_).ravel()
col_names = []
for col in categorical_feature:
  for val in train_data[col].unique():
    col_names.append("{}_{}".format(col, val))
oneHot_X=pd.DataFrame(data = matrix, columns=col_names, dtype=int)

not_ontHot_X_columns=list()
for column in X.columns.tolist():
    if column not in categorical_feature:
        not_ontHot_X_columns.append(column)
not_oneHot_X=X[not_ontHot_X_columns]

X=pd.concat([oneHot_X,not_oneHot_X],axis=1)
X_class_one=X.loc[X['y']==1]
X_class_zore=X.loc[X['y']==0]
X_class_one=pd.concat([X_class_one,X_class_one,X_class_one,X_class_one])
X=pd.concat([X_class_zore,X_class_one])

def train_dev_split(X,dev_ratio=0.25):
    X.sample(frac=1)#打乱
    train_size=int(X.shape[0]*(1-dev_ratio))#分割训练集与检验集
    return X[:train_size],X[train_size:]

X_train,X_dev=train_dev_split(X)

X_train_y=X_train.iloc[:,-1]
X_train=X_train.iloc[:,:-1]
data_dim=X_train.shape[1]
train_nums=X_train.shape[0]
X_train=X_train.values

X_dev_y=X_dev.iloc[:,-1]
X_dev=X_dev.iloc[:,:-1]
dev_nums=X_dev.shape[0]
X_dev=X_dev.values


def sigmoid(z):
    return np.clip(1/1+np.exp(-z),1e-8, 1 - (1e-8))

def f(x,w,b):
    return sigmoid(np.dot(x,w)+b)

def predict(x,w,b):
    return np.round(f(x,w,b)).astype(np.int)

def accuracy(Y_pred,Y_label):
    return 1-np.mean(np.abs(Y_pred-Y_label))

def cross_entropy_loss(y_pred,Y_label):
    return -np.dot(Y_label,np.log(y_pred))-np.dot((1-Y_label),np.log(1-y_pred))

def gradient(x,Y_label,w,b):
    # print(type(x))
    # print('x'+str(x.shape))
    # print('w'+str(w.shape))
    # print('b'+str(b.shape))
    y_pred=f(x,w,b)
    pred_error=Y_label - y_pred
    pred_error=np.array([pred_error.tolist()])
    w_grad=-np.sum(pred_error*x.T,axis=1)
    b_grad=-np.sum(pred_error)
    return w_grad,b_grad

w=np.zeros((data_dim,))
b=np.zeros((1,))

epoch_nums=10
batch_size=8
learning_rate=0.2

train_loss=[]
dev_loss=[]
train_acc=[]
dev_acc=[]

step = 1

for epoch in range(epoch_nums):
    for train_index in range(int(np.floor(train_nums/batch_size))):
        print('epoch:'+str(epoch)+'step:' + str(step)+'/'+str(int(np.floor(train_nums/batch_size))))
        X=X_train[train_index*batch_size:(train_index+1)*batch_size]
        Y=X_train_y[train_index*batch_size:(train_index+1)*batch_size]

        w_grad,b_grad=gradient(X,Y,w,b)

        w=w-learning_rate/np.sqrt(step)*w_grad
        b=b-learning_rate/np.sqrt(step)*b_grad

        step+=1

        y_train_pred=f(X_train,w,b)
        Y_train_pred=np.round(y_train_pred)
        train_acc.append(accuracy(Y_train_pred,X_train_y))
        train_loss.append(cross_entropy_loss(y_train_pred,X_train_y)/train_nums)
        print(train_acc[-1])
        print(train_loss[-1])
        y_dev_pred=f(X_dev,w,b)
        Y_dev_pred=np.round(y_dev_pred)
        dev_acc.append(accuracy(Y_dev_pred,X_dev_y))
        dev_loss.append(cross_entropy_loss(y_dev_pred,X_dev_y)/dev_nums)

print('Training loss: {}'.format(train_loss[-1]))
print('Development loss: {}'.format(dev_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Development accuracy: {}'.format(dev_acc[-1]))


