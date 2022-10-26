#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 20:05:55 2022

@author: apple1
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
data=pd.read_excel('E Commerce Dataset.xlsx',sheet_name= 'E Comm')

#taking numerical columns
num_e=data.select_dtypes(exclude='object').columns
num=data.select_dtypes(include=["int64","float64"])
num=num.iloc[:,1:]

#taking catagorical columns
category=data.select_dtypes(include=["object"])



# Replace CC to CreditCard
data['PreferredPaymentMode'] = data['PreferredPaymentMode'].replace({'CC':'Credit Card'})

# Replace COD to  Cash On Delivery
data['PreferredPaymentMode'] = data['PreferredPaymentMode'].replace({'COD':'Cash on Delivery'})

# Replace Mobile to Mobile Phone
data['PreferedOrderCat'] = data['PreferedOrderCat'].replace({'Mobile':'Mobile Phone'})

# Replace Phone to Mobile Phone
data['PreferredLoginDevice'] = data['PreferredLoginDevice'].replace({'Phone':'Mobile Phone'})

# Handling Missing Values

# Tenure
mask1 = (data['Tenure'].isna()) & (data['Churn']==1) # fill 1
mask0 = (data['Tenure'].isna()) & (data['Churn']==0)  # fill 10
data.loc[mask1,'Tenure'] = data[mask1]['Tenure'].fillna(1)
data.loc[mask0,'Tenure'] = data[mask0]['Tenure'].fillna(10)

# Warehouse to home
mask0 = (data['WarehouseToHome'].isna()) & (data['Churn']==0)  # fill 13
mask1 = (data['WarehouseToHome'].isna()) & (data['Churn']==1) # fill 15
data.loc[mask0,'WarehouseToHome'] = data[mask0]['WarehouseToHome'].fillna(13)
data.loc[mask1,'WarehouseToHome'] = data[mask1]['WarehouseToHome'].fillna(15)

#HourSpend on App
data ['HourSpendOnApp'] = data['HourSpendOnApp'].fillna(3)
data ['HourSpendOnApp'].isna().sum()

#Order Amount Hike From Last Year
data['OrderAmountHikeFromlastYear'] =data['OrderAmountHikeFromlastYear'].fillna(14.5)

# Coupon Used
data['CouponUsed'] = data['CouponUsed'].fillna(1)

# Order Count
data['OrderCount']=data['OrderCount'].fillna(2)

# Day Since Last Order
mask0 = (data['DaySinceLastOrder'].isna()) & (data['Churn']==0)  # fill 4
mask1 = (data['DaySinceLastOrder'].isna()) & (data['Churn']==1) # fill 2
data.loc[mask0,'DaySinceLastOrder'] = data[mask0]['DaySinceLastOrder'].fillna(4)
data.loc[mask1,'DaySinceLastOrder'] = data[mask1]['DaySinceLastOrder'].fillna(2)

# Handling Outlier

# Tenure
data['Tenure'] = np.where(data['Tenure'] > 30,30,data['Tenure'])
q1,q3 = np.percentile(data['WarehouseToHome'],[25,75])
iqr = q3-1
ul = q3 = 1.5*iqr
ul , np.percentile(data['WarehouseToHome'],99)

# Warehouse to Home
data['WarehouseToHome'] = np.where(data['WarehouseToHome'] > 36, 36,data['WarehouseToHome'])

# Encoding
from sklearn.preprocessing import OneHotEncoder
enc_drop = OneHotEncoder(drop= 'first')
enc_drop.fit(category)
encoded = enc_drop.transform(category).toarray()

#enc.inverse_transform(encoded)
df_enc = data.join(pd.DataFrame(encoded,columns =['PreferredLoginDevice_Mobile Phone',
       'PreferredPaymentMode_Credit Card',
       'PreferredPaymentMode_Debit Card', 'PreferredPaymentMode_E wallet',
       'PreferredPaymentMode_UPI', 'Gender_Male',
       'PreferedOrderCat_Grocery', 'PreferedOrderCat_Laptop & Accessory',
       'PreferedOrderCat_Mobile Phone', 'PreferedOrderCat_Others',
       'MaritalStatus_Married', 'MaritalStatus_Single'] ))

df_enc =df_enc.drop(['PreferredLoginDevice','PreferredPaymentMode', 'Gender',
       'PreferedOrderCat','MaritalStatus'],axis=1)

# New column Tenure in year
df_enc['Tenure_year'] = df_enc['Tenure']/12

# Data Scaling



# X
x = df_enc.drop(['Churn', 'CustomerID'], axis =1)
y = df_enc['Churn']

from imblearn import over_sampling
x,y = over_sampling.SMOTE(0.5).fit_resample(x,y)


from xgboost import XGBClassifier

xgb = XGBClassifier(eval_metric='error')

# Splitting data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

# Traning data
xgb.fit(x_train,y_train)
pred= xgb.predict(x_test)
pickle.dump(xgb,open('model.pkl','wb'))

