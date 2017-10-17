# -*- coding: utf-8 -*-

day_time = '_03_12_1'

import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.svm import SVR
import sys
import numpy as np

def calculate_score(pre,real):
    if(len(pre.shape)==1):
        pre = DataFrame(pre,columns=[0])
        real = DataFrame(real,columns=[0])
    else:
        pre = DataFrame(pre,columns=[i for i in range(pre.shape[1])])
        real = DataFrame(real,columns=[i for i in range(real.shape[1])])        
        
    if(len(pre)!=len(real)):
        print ('len(pre)!=len(real)','\n')
    if(len(pre.columns)!=len(real.columns)):
        print ('len(pre.columns)!=len(real.columns)','\n')
    N = len(pre)    #N：商家总数
    T = len(pre.columns)    
    print ('N:',N,'\t','T:',T,'\n')
    
    n = 0
    t = 0
    L=0
    
    while(t<T):
        n=0
        while(n<N):
            c_it = round(pre.ix[n,t])       #c_it：预测的客流量
            c_git = round(real.ix[n,t])    #c_git：实际的客流量
            
            
            if((c_it==0 and c_git==0) or (c_it+c_git)==0 ):
                c_it=1
                c_git=1
            
            L = L+abs((float(c_it)-c_git)/(c_it+c_git))
            n=n+1
        t=t+1
    return L/(N*T)

def get_result(result):
    if(len(result.shape)==1):
        df = DataFrame(result,columns=[0])
    else:
        df = DataFrame(result,columns=['col_'+str(i) for i in range(result.shape[1])])
    df.insert(0,'shop_id',[i for i in range(1,2001)])
    return df.drop('shop_id',axis=1)

train_x = pd.read_csv('train_1/train_x'+day_time+'.csv')
train_y = pd.read_csv('train_1/train_y'+day_time+'.csv')
test_x = pd.read_csv('test_1/test_x'+day_time+'.csv')
test_y = pd.read_csv('test_1/test_y'+day_time+'.csv')

x_scaled_train = preprocessing.scale(train_x)
x_scaled_test = preprocessing.scale(test_x)


result = DataFrame()

for i in range(0,7):
    
    svr = SVR(kernel='linear', C=1.0, tol=0.01,epsilon=0.1)
    svr.fit(x_scaled_train,train_y.icol(i))
    pre = (svr.predict(x_scaled_test)).round()
    result['col'+str(i)] = pre


result = get_result(result.values)
result.to_csv('SVRresult'+day_time+'.csv',index=False,header=False)

result=np.asarray(result)
result=result.astype(int)

score = calculate_score(result,test_y.values)
print (score)