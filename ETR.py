# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import random
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
import sys

day_time = '_03_12_1'

train_x = pd.read_csv('train_1/train_x'+day_time+'.csv')
train_y = pd.read_csv('train_1/train_y'+day_time+'.csv')
test_x = pd.read_csv('test_1/test_x'+day_time+'.csv')
test_y = pd.read_csv('test_1/test_y'+day_time+'.csv')

def get_result(result):
    if(len(result.shape)==1):
        df = DataFrame(result,columns=[0])
    else:
        df = DataFrame(result,columns=['col_'+str(i) for i in range(result.shape[1])])
    df.insert(0,'shop_id',[i for i in range(1,2001)])
    # df = pd.merge(df,df,on='shop_id')
    return df    

def ftoi(a):
    for i in range(len(a)):
        for j in range(len(a[0])):
            a[i,j] = int(a[i,j])
    return a


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
            c_it = round(pre.ix[n,t])       
            c_git = round(real.ix[n,t])          
            
            if((c_it==0 and c_git==0) or (c_it+c_git)==0 ):
                c_it=1
                c_git=1
            
            L = L+abs((float(c_it)-c_git)/(c_it+c_git))
            n=n+1
        t=t+1
    #print L
    return L/(N*T)


def draw_feature_importance(train_x,clf):
    feature_names = train_x.columns
    feature_importance = clf.feature_importances_
    df = DataFrame({'feature_names':feature_names,'feature_importances':feature_importance})
    df1 = df.sort(columns='feature_importances',ascending=False)
    df1.index = [i for i in range(len(df1))]
    fig = plt.figure(num=random.randint(1,10000))
    ax = fig.add_subplot(111) 
        
    ax.set_xticks([i for i in range(len(df.feature_names))]) 
    ax.set_xticklabels(df1.feature_names,rotation=-90)
    ax.grid()
    ax.plot(df1.feature_importances,label='feature_importance')
    plt.subplots_adjust(bottom=0.2)
    return df1



def cal(i):
    ET = ExtraTreesRegressor(n_estimators=1300,random_state=1,n_jobs=-1,
        min_samples_split=2,min_samples_leaf=2,max_depth=i,max_features=78)
    ET.fit(train_x,train_y)
    pre = (ET.predict(test_x)).round()
    pre1 = ftoi(pre)
    result = get_result(pre1)
    result.to_csv('ETRresult'+day_time+'.csv',index=False,header=False)
    pre=np.asarray(pre)
    pre=pre.astype(int)


    score = calculate_score(pre,test_y.values)
    print('max_depth',i,"     ",score)
    return pre
    print (score)


max_depth = [24,25]


for i in max_depth:
    pre = cal(i)