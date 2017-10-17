# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd
import sys
from pandas import DataFrame
from sklearn.preprocessing import PolynomialFeatures
from tool import *
from numpy import genfromtxt

def every_shop_open_ratio(threshold=0,start_day=464,end_day=495,smaller=False):
    f_read=open('data/shop_flow.pkl','rb')
    shop_flow_matrix=pickle.load(f_read)
    Open_ratios = []
    for shop_id in range(1,2001):
        single_row = 0.0
        for i in range(start_day,end_day):
            if shop_flow_matrix[shop_id][i] >0:
                single_row+=1
        open_ratio_ = float(single_row/(end_day-start_day))
        Open_ratios.append(open_ratio_)
   
    Open_ratios = np.asarray(Open_ratios)
    df = DataFrame({'shop_id':np.arange(1,2001),'open_ratio':Open_ratios})
    return  df  

f_read=open('data/shop_flow.pkl','rb')
shop_flow_matrix=pickle.load(f_read)

train_weekend = [457,458,464,465,471,472,478,479]
train_ratio_wk = shop_flow_matrix[1:,train_weekend[0]]
for i in range(1,len(train_weekend)):
    train_ratio_wk += shop_flow_matrix[1:,train_weekend[i]]
train_ratio_wk = train_ratio_wk / 2.0

test_weekend = [464,465,471,472,478,479,485,486]
test_ratio_wk = shop_flow_matrix[1:,test_weekend[0]]
for i in range(1,len(test_weekend)):
    test_ratio_wk += shop_flow_matrix[1:,test_weekend[i]]
test_ratio_wk = test_ratio_wk / 2.0


day_time = '_03_12_1'

weekA = pd.read_csv('data/pay/weekA.csv'); weekA.index=weekA.shop_id
weekB = pd.read_csv('data/pay/weekB.csv'); weekB.index=weekB.shop_id
weekC = pd.read_csv('data/pay/weekC.csv'); weekC.index=weekC.shop_id
weekD = pd.read_csv('data/pay/weekD.csv'); weekD.index=weekD.shop_id

weekA_weather = pd.read_csv('data/weather/weekA_weather.csv'); weekA_weather.index = weekA_weather.shop_id
weekB_weather = pd.read_csv('data/weather/weekB_weather.csv'); weekB_weather.index = weekB_weather.shop_id
weekC_weather = pd.read_csv('data/weather/weekC_weather.csv'); weekC_weather.index = weekC_weather.shop_id
weekD_weather = pd.read_csv('data/weather/weekD_weather.csv'); weekD_weather.index = weekD_weather.shop_id

weekA_view = pd.read_csv('data/view/weekA_view.csv'); weekA_view.index = weekA_view.shop_id
weekB_view = pd.read_csv('data/view/weekB_view.csv'); weekB_view.index = weekB_view.shop_id
weekC_view = pd.read_csv('data/view/weekC_view.csv'); weekC_view.index = weekC_view.shop_id
weekD_view = pd.read_csv('data/view/weekD_view.csv'); weekD_view.index = weekD_view.shop_id

train_vary_rate = cal_varyday(end = 495-14)

# train_vary3month_rate = cal_varyday(start =402-14 ,end = 495-14)

train_vary_week_rate = cal_vary_week_rate(end = 21)
train_vary_week3month_rate = cal_vary_week3month_rate(end = 21)
train_holiday_sale = cal_holiday_sale(end = 490-14)
train_holiday3month_sale = cal_holiday_sale(start=398-14,end=490-14)
economy = genfromtxt('data/economic.csv',delimiter=',')
train_ave_weekday3month = cal_ave_weekday3month(begin = 403-14,end =494-14)






train_x_weather = pd.merge(weekA_weather,weekB_weather,on='shop_id').drop('shop_id',axis=1)
train_sunny_rate = train_x_weather.sum(axis=1)
train_x_view = (pd.merge(weekB_view,weekC_view,on='shop_id')).drop('shop_id',axis=1)
'''  poly   degree=1     '''
poly = PolynomialFeatures(1,interaction_only=True,include_bias=False)
train_x = pd.merge(weekA,weekB,on='shop_id').drop('shop_id',axis=1)                                       #train = weekA+ weekB + weekC


train_sum = train_x.sum(axis=1)
train_mean = train_x.mean(axis=1)

train_open_ratio_A = every_shop_open_ratio(start_day=460,end_day=474)
train_open_ratio_B = every_shop_open_ratio(start_day=474,end_day=481)
train_open_ratio = (train_open_ratio_A.open_ratio * 2 + train_open_ratio_B.open_ratio)/3
train_ratio_wk = train_ratio_wk/(train_sum.replace(0,1))

train_std = train_x.std(axis=1)
train_max = train_x.max(axis=1)
train_min = train_x.min(axis=1)
train_median = train_x.median(axis=1)
train_mad = train_x.mad(axis=1)
train_var = train_x.var(axis=1) 

OHE_cate_1 = transfrom_Arr_DF(make_OHE(cate_1_num),'cate_1_')

OHE_cate_2 = transfrom_Arr_DF(make_OHE(cate_2_num),'cate_2_')

OHE_cate_3 = transfrom_Arr_DF(make_OHE(cate_3_num),'cate_3_')

OHE_score = transfrom_Arr_DF(make_OHE(score),'shop_info_score_')

OHE_shop_level = transfrom_Arr_DF(make_OHE(level),'shop_info_level_')

train_x = transfrom_Arr_DF(poly.fit_transform(train_x))
train_x['sumABCD'] = train_sum
train_x['open_ratio'] = train_open_ratio
train_x['ratio_wk'] = train_ratio_wk
train_x['meanABCD'] = train_mean

train_x['vary_rate'] = train_vary_rate[:,1]
train_x['vary_week_rate'] = train_vary_week_rate[:,1]
train_x['vary_week3month_rate'] = train_vary_week3month_rate[:,1]
train_x['holiday_sale'] = train_holiday_sale[:,1]
train_x['holiday3month_sale'] = train_holiday3month_sale[:,1]
train_x['economy'] = economy


train_x['comment_cnt'] = comment_cnt
train_x['per_pay'] = per_pay

train_x['Monday'] = train_ave_weekday3month[:,1]
train_x['Tues'] = train_ave_weekday3month[:,2]
train_x['Wends'] = train_ave_weekday3month[:,3]
train_x['Thurs'] = train_ave_weekday3month[:,4]
train_x['Friday'] = train_ave_weekday3month[:,5]

train_x = train_x.join(OHE_cate_1,how='left')
train_x = train_x.join(OHE_cate_2,how='left')
train_x = train_x.join(OHE_cate_3,how='left')

train_x['location_id'] = location_id
train_x = train_x.join(OHE_shop_level,how='left')
train_x = train_x.join(train_x_view)

train_x['std'] = train_std
train_x['max'] = train_max
train_x['min'] = train_min
train_x['median'] = train_median
train_x['mad'] = train_mad
train_x['var'] = train_var

#train_x = train_x.join(train_x_weather,how='left')

train_y = weekC.drop('shop_id',axis=1)

train_x.to_csv('train_1/train_x'+day_time+'.csv',index=False)
train_y.to_csv('train_1/train_y'+day_time+'.csv',index=False)
#--------------------------------------------------------------------------------------------------------------------------------------------------------
test_x = pd.merge(weekB,weekC,on='shop_id').drop('shop_id',axis=1)                                      #test = weekB + weekC + weekD 
test_x_weather = pd.merge(weekB_weather,weekC_weather,on='shop_id').drop('shop_id',axis=1)
test_sunny_rate = test_x_weather.sum(axis=1)

test_x_view = pd.merge(weekB_view,weekC_view,on='shop_id').drop('shop_id',axis=1)

test_vary_rate = cal_varyday(end = 495-7)

# test_vary3month_rate = cal_varyday(start =402-7 ,end = 495-7)   #useless

test_vary_week_rate = cal_vary_week_rate(end = 14)
test_vary_week3month_rate = cal_vary_week3month_rate(end = 14)
test_holiday_sale = cal_holiday_sale(end = 490-7)
test_holiday3month_sale = cal_holiday_sale(start=398-7,end = 490-7)
test_ave_weekday3month = cal_ave_weekday3month(begin = 403-7,end =494-7)

test_sum = test_x.sum(axis=1)
test_mean = test_x.mean(axis=1)
test_open_ratio = every_shop_open_ratio(start_day=467,end_day = 488)



test_ratio_wk = test_ratio_wk/(test_sum.replace(0,1))

test_std = test_x.std(axis=1)
test_max = test_x.max(axis=1)
test_min = test_x.min(axis=1)
test_median = test_x.median(axis=1)
test_mad = test_x.mad(axis=1)
test_var = test_x.var(axis=1)

test_x = transfrom_Arr_DF(poly.fit_transform(test_x))

test_x['sumABCD'] = test_sum
test_x['open_ratio'] = test_open_ratio.open_ratio
test_x['ratio_wk'] = test_ratio_wk
test_x['meanABCD'] = test_mean

test_x['vary_rate'] = test_vary_rate[:,1]

# test_x['vary3month_rate'] = test_vary3month_rate[:,1]

test_x['vary_week_rate'] = test_vary_week_rate[:,1]
test_x['vary_week3month_rate'] = test_vary_week3month_rate[:,1]
test_x['holiday_sale'] = test_holiday_sale[:,1]
test_x['holiday3month_sale'] = test_holiday3month_sale[:,1]
test_x['economy'] = economy

test_x['comment_cnt'] = comment_cnt
test_x['per_pay'] = per_pay

test_x['Monday'] = test_ave_weekday3month[:,1]
test_x['Tues'] = test_ave_weekday3month[:,2]
test_x['Wends'] = test_ave_weekday3month[:,3]
test_x['Thurs'] = test_ave_weekday3month[:,4]
test_x['Friday'] = test_ave_weekday3month[:,5]



test_x = test_x.join(OHE_cate_1,how='left')
test_x = test_x.join(OHE_cate_2,how='left')
test_x = test_x.join(OHE_cate_3,how='left')

test_x['location_id'] = location_id
test_x = test_x.join(OHE_shop_level,how='left')
test_x = test_x.join(test_x_view)
test_x['std'] = test_std
test_x['max'] = test_max
test_x['min'] = test_min
test_x['median'] = test_median
test_x['mad'] = test_mad
test_x['var'] = test_var

#test_x = test_x.join(test_x_weather,how='left')

test_y = weekD.drop('shop_id',axis=1)

test_x.to_csv('test_1/test_x'+day_time+'.csv',index=False)
test_y.to_csv('test_1/test_y'+day_time+'.csv',index=False)


i=0
for f in test_x.columns:
    i+=1
print(i)
