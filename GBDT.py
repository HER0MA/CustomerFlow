# -*- coding: utf-8 -*-

day_time = '_03_12_1'

import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import sys
import numpy as np
sys.path.append('../tools')

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


param = {'subsample':[0.7,0.7,1,1,1,1,1],'min_samples_leaf':[1,1,1,1,1,1,1],'n_estimators':[200,200,200,200,200,200,100],'min_samples_split':[6,4,2,8,2,4,4],\
        'learning_rate':[0.05,0.05,0.05,0.05,0.05,0.05,0.1],'max_features':[78,78,78,78,78,78,78],'random_state':[1,1,1,1,1,1,1]\
        ,'max_depth':[4,4,4,4,4,4,4]}


result = DataFrame()
for i in range(0,7):
    GB = GradientBoostingRegressor(n_estimators=param['n_estimators'][i],learning_rate=0.05,random_state=1,\
                                min_samples_split=param['min_samples_split'][i],min_samples_leaf=1,max_depth=param['max_depth'][i],max_features=param['max_features'][i],subsample=0.85)     
   
    GB.fit(train_x,train_y.icol(i))
    pre = (GB.predict(test_x)).round()
    
    result['col'+str(i)] = pre


result = get_result(result.values)
result.to_csv('GBDTresult'+day_time+'.csv',index=False,header=False)

result=np.asarray(result)
result=result.astype(int)

score = calculate_score(result,test_y.values)
print (score)