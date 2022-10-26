from flask import Flask, request, render_template
import numpy as np
import pandas as pd
#import xgboost as xgb

import joblib
model = joblib.load('xgb_model.sav')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=[ 'POST'])
def prediction():
    df={}
    df['Tenure'] = [int(request.values['Tenure'])]
    
    
    df['PreferredLoginDevice'] = [request.values['PreferredLoginDevice']]
    df['CityTier'] = [int(request.values['CityTier'])]
    df['WarehouseToHome'] = [float(request.values['WarehouseToHome'])]
   
    df['PreferredPaymentMode'] = [request.values['PreferredPaymentMode']]
    df['Gender'] = [request.values['Gender']]
    df['HoursSpendOnApp'] = [int(request.values['HoursSpendOnApp'])]
    df['NumberOfDeviceRegistered'] = [int(request.values['NumberOfDeviceRegistered'])]
    df['PreferedOrderCat'] = [request.values['PreferedOrderCat']]
    df['SatisfactionScore'] = [int(request.values['SatisfactionScore'])]
    df['MaritalStatus'] = [request.values['MaritalStatus']]
    df['NumberOfAddress'] = [int(request.values['NumberOfAddress'])]
    df['Complain'] = [int(request.values['Complain'])]
    df['OrderAmountHikeFromlastYear'] = [float(request.values['OrderAmountHikeFromlastYear'])]
    df['CouponUsed'] = [int(request.values['CouponUsed'])]
    df['OrderCount'] = [int(request.values['OrderCount'])]
    df['DaySinceLastOrder'] = [int(request.values['DaySinceLastOrder'])]
    df['CashbackAmount'] = [float(request.values['CashbackAmount'])]
    df = pd.DataFrame(df)
    #df['Tenure'] = np.where(df['Tenure'] > 30,30,df['Tenure'])
    #df['WarehouseToHome'] = np.where(df['WarehouseToHome'] > 36, 36,df['WarehouseToHome'])
    import joblib
    ohe = joblib.load('oneHotEnc.sav')
    # apply one hot to data, and join encoded columns with data
    category= ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']
    
    encoded = ohe.transform(df[category]).toarray()
    
    df_enc = df.join(pd.DataFrame(encoded,columns =['PreferredLoginDevice_Mobile Phone',
       'PreferredPaymentMode_Credit Card',
       'PreferredPaymentMode_Debit Card', 'PreferredPaymentMode_E wallet',
       'PreferredPaymentMode_UPI', 'Gender_Male',
       'PreferedOrderCat_Grocery', 'PreferedOrderCat_Laptop & Accessory',
       'PreferedOrderCat_Mobile Phone', 'PreferedOrderCat_Others',
       'MaritalStatus_Married', 'MaritalStatus_Single'] ))
    
    df_enc =df_enc.drop(['PreferredLoginDevice','PreferredPaymentMode', 'Gender',
       'PreferedOrderCat','MaritalStatus'],axis=1)
    df_enc['Tenure_year'] = df_enc['Tenure']/12
    

    prediction=model.predict(df_enc)


    return render_template("prediction.html",prediction_text='Churn Score is {}'.format(prediction))
    print(prediction)
   
                           
    

if __name__=="__main__":
    app.run()
