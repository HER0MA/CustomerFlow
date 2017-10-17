# -*- coding: utf-8 -*-
import pickle
import numpy as np 
import pandas as pd 
import csv
from pandas import Series,DataFrame
from collections import defaultdict
from sklearn import preprocessing  
from numpy import genfromtxt

def transfrom_Arr_DF(arr,col_name = 'col_'):
    if(len(arr.shape)==1):
        df = DataFrame(arr,columns=['col_0'])
    else:
        df = DataFrame(arr,columns=[col_name+str(i) for i in range(arr.shape[1])])
    return df

def make_OHE(names):
    data = []
    for name in names:
        data.append([name])          
    enc = preprocessing.OneHotEncoder()
    enc.fit(data)
    OHE_data = enc.transform(data).toarray()  
    return OHE_data


data_train = pd.read_csv('shop.txt')
data_train.index=data_train.shop_id
    
data_train = data_train.fillna(0)

city_name = data_train['city_name']
location_id = data_train['location_id'].values
per_pay = data_train['per_pay'].values


num2chi=[]
chi2num = dict()

shop_number = pd.read_csv('data/city_shop_number.csv')

for num in range(shop_number.shape[0]):
    chi2num[shop_number['city_name'][num]] = shop_number['encode'][num]

city_num=[]
for shop_id in range(1,2001):
    city_num.append(chi2num[city_name[shop_id]])



cate_1_num=[]
cate_1_name = data_train['cate_1_name']
for shop_id in range(1,2001):
    if cate_1_name[shop_id] == '美食':
        cate_1_num.append(0)
    elif cate_1_name[shop_id] == '超市便利店':
        cate_1_num.append(1)
    else:
        cate_1_num.append(100)


cate_2_num=[]
cate_2_dict={'快餐':4, '超市':1, '便利店':9, '休闲茶饮':0, '小吃':5,  '休闲食品':2, '烘焙糕点':3,  '中餐':6,  '其他美食':10,
'火锅':7}
cate_2_name = data_train['cate_2_name']
for shop_id in range(1,2001):
    if cate_2_name[shop_id] in cate_2_dict.keys():
        cate_2_num.append(cate_2_dict[cate_2_name[shop_id]]) 
    else:
        cate_2_num.append(100)


cate_3_num=[]
cate_3_dict={'西式快餐':5 ,'中式快餐':8 ,'生鲜水果':3 ,'奶茶':2 ,'其他小吃': 6,'面包': 4 , '饮品/甜点':0,
'面点':16,'蛋糕':11}
cate_3_name = data_train['cate_3_name']

for shop_id in range(1,2001):
    if cate_3_name[shop_id] in cate_3_dict.keys():
        cate_3_num.append(cate_3_dict[cate_3_name[shop_id]]) 
    else:
        cate_3_num.append(100)


score=data_train['score'].values


comment_cnt = data_train['comment_cnt'].values


level = data_train['shop_level'].values





def cal_open_lenth():
    f_read=open('data/shop_flow.pkl','rb')
    shop_flow_matrix=pickle.load(f_read)
    open_lenth = np.zeros([2000,2],dtype=np.int32)
    for shop_id in range(2000):
        open_lenth[shop_id][0]=shop_id+1
        index=1
        while index<495:
            if shop_flow_matrix[shop_id+1][index]==0:
                index+=1
            elif shop_flow_matrix[shop_id+1][index]>0:
                break
        open_lenth[shop_id][1]=495-index
    return open_lenth

open_lenth = cal_open_lenth()


def cal_varyday(start=2,end = 495):
    f_read=open('data/shop_flow.pkl','rb')
    shop_flow_matrix=pickle.load(f_read)
    vary_rate = np.zeros([2000,2],dtype=np.int32)
    for shop_id in range(2000):
        vary_rate[shop_id][0]=shop_id+1
        buf = 0
        count = 0
        for j in range(start,end):
            buf+=abs(shop_flow_matrix[shop_id+1][j]-shop_flow_matrix[shop_id+1][j-1])
            if shop_flow_matrix[shop_id+1][j] >0 and shop_flow_matrix[shop_id+1][j-1]>0:
                count+=1
        if count ==0:
            count = 1
        vary_rate[shop_id][1]=int(buf/count)
    return vary_rate


def cal_vary_week_rate(end = 7):
    f_read=open('data/shop_flow.pkl','rb')
    shop_flow_matrix=pickle.load(f_read)
    vary_week_rate = np.zeros([2000,2],dtype=np.int32)
    for shop_id in range(2000):
        vary_week_rate[shop_id][0]=shop_id+1
        index = end+7
        count = 0
        buf = 0
        if end == 7:
            buf = abs(shop_flow_matrix[shop_id+1][-index:-end].sum()-shop_flow_matrix[shop_id+1,-end:].sum())
            count = 1        
        while  index<=open_lenth[shop_id][1]-7:
            buf += abs(shop_flow_matrix[shop_id+1][-(index+7):-index].sum()-shop_flow_matrix[shop_id+1,-index:-(index-7)].sum())
            if shop_flow_matrix[shop_id+1][-(index+7):-index].sum() >0 and shop_flow_matrix[shop_id+1,-index:-(index-7)].sum()>0:
                count += 1
            index += 7
        vary_week_rate[shop_id][1] = buf/count
    return vary_week_rate

def cal_vary_week3month_rate(end = 7):
    f_read=open('data/shop_flow.pkl','rb')
    shop_flow_matrix=pickle.load(f_read)
    vary_week_rate = np.zeros([2000,2],dtype=np.int32)
    for shop_id in range(2000):
        vary_week_rate[shop_id][0]=shop_id+1
        index = end+7
        count = 0
        buf = 0
        shut_index = min(open_lenth[shop_id][1]-7,91)
        if end == 7:
            buf = abs(shop_flow_matrix[shop_id+1][-index:-end].sum()-shop_flow_matrix[shop_id+1,-end:].sum())
            count = 1        
        while  index<=shut_index:
            buf += abs(shop_flow_matrix[shop_id+1][-(index+7):-index].sum()-shop_flow_matrix[shop_id+1,-index:-(index-7)].sum())
            if shop_flow_matrix[shop_id+1][-(index+7):-index].sum() >0 and shop_flow_matrix[shop_id+1,-index:-(index-7)].sum()>0:
                count += 1
            index += 7
        if count == 0:
            # print(shop_id,"    ",index)
            count=1
        vary_week_rate[shop_id][1] = buf/count
    return vary_week_rate

def cal_holiday_sale(start=1,end = 490):
    f_read=open('data/shop_flow.pkl','rb')
    shop_flow_matrix=pickle.load(f_read)
    mat = genfromtxt('data/holiday.csv',delimiter=',')
    holiday_index = []
    for i in range(start,end):
        if mat[i-1][1]==1:
            holiday_index.append(i+5)
    
    holiday_sale = np.zeros([2000,2],dtype=np.int32)
    for shop_id in range(2000):
        count = 0
        buf = 0
        holiday_sale[shop_id][0]=shop_id+1
        for i in holiday_index:
            if shop_flow_matrix[shop_id+1][i]>0:
                count += 1
                buf += shop_flow_matrix[shop_id+1][i]
        if count == 0:
            # print(shop_id)
            count = 1
        holiday_sale[shop_id][1] = buf/count
    return holiday_sale




def cal_ave_weekday3month(begin,end):
    ave_weekday3month = np.zeros([2000,8],dtype=np.int32)
    f_read=open('data/shop_flow.pkl','rb')
    shop_flow_matrix=pickle.load(f_read)
    for weekday in range(1,8):
        for shop_id in range(2000):
            index = begin
            ave_weekday3month[shop_id][0]=shop_id+1
            buf = 0
            count = 0
            while  index<=end:
                buf += shop_flow_matrix[shop_id+1][index]
                if shop_flow_matrix[shop_id+1][index] > 0 :
                    count += 1
                index += 7
            if count ==0:
                # print(shop_id)
                count =1
            ave_weekday3month[shop_id][weekday] = buf/count
    return ave_weekday3month

