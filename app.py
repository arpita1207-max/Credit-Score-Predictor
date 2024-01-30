import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import re

#Function Transformers
def transform_payment_of_min_amount_column(column):
    return column.replace({'Yes': 1, 'No': 0, 'NM': 2}).astype('float')

def transform_payment_behaviour(column):
    return column.replace({'Low_spent_Small_value_payments':1, 'Low_spent_Medium_value_payments':2,
    'Low_spent_Large_value_payments':3, 'High_spent_Small_value_payments':4,
    'High_spent_Medium_value_payments':5, 'High_spent_Large_value_payments':6}).astype('float')

def transform_month(column):
    return column.replace({
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }).astype('float')
    
    
def credit_history(x):
    if pd.isna(x):
        return np.nan  # Return NaN if the input is NaN
    else:
        x = re.split('\s+', x)
        x[0] = int(x[0]) * 12
        return x[0] + int(x[3])

st.title('Credit Score Classifier')
data=pd.read_csv(r'data\df.csv')
data=data.drop(columns='Unnamed: 0')
model=joblib.load('pipeline1.pkl')

col1,col2=st.columns(2)
cust_ids='CUS_'+data['Customer_ID'].astype('int').apply(lambda x: hex(x))
with col1:
    Customer_ID=st.selectbox("Select the Customer ID",(cust_ids.value_counts().index.tolist()))
    Customer_ID=float(int(Customer_ID[4:],16))
with col2:
    Occupation=st.selectbox("Select the Occupation",(data['Occupation'].value_counts().index.tolist()))

col1,col2=st.columns(2)
with col1:
    month_name=['January', 'Feburary', 'March', 'April', 'May', 'June',
                'July', 'August','September','October','November','December']

    Month=(st.selectbox("Select the Month",month_name))
    
with col2:
    Monthly_Inhand_Salary=float(st.number_input("Enter the Monthly Inhand Salary"))

col1,col2=st.columns(2)
with col1:
    Num_Bank_Accounts=float(st.number_input("Enter the number of accounts"))
with col2:
    Num_Credit_Card=float(st.number_input("Enter the number of credit cards"))

col1,col2=st.columns(2)
with col1:
    Num_of_Loan=float(st.number_input("Enter the number of loans"))
with col2:
    Outstanding_Debt=float(st.number_input("Enter the outstanding debt"))
   
col1,col2=st.columns(2)
with col1:
    Changed_Credit_Limit=float(st.number_input('Enter the changed credit limit'))
with col2:
    Age=float(st.number_input('Enter the age'))
    
col1,col2=st.columns(2)
with col1:
    payment_behaviour=['Low_spent_Small_value_payments', 'Low_spent_Medium_value_payments',
    'Low_spent_Large_value_payments', 'High_spent_Small_value_payments',
    'High_spent_Medium_value_payments', 'High_spent_Large_value_payments']
    Payment_Behaviour=(st.selectbox("Select the payment behaviour",payment_behaviour))
with col2:
    payment_min_amount=['Yes','No','NM']
    Payment_of_Min_Amount=(st.selectbox("Select the payment minimum amount",payment_min_amount))

col1,col2,col3=st.columns(3)
with col1:
    Credit_Years=(st.number_input('Enter the years'))
with col2:
    Credit_Months=(st.number_input('Enter the months'))
with col3:
    Delay_from_due_date=st.number_input('Enter days delayed from due date')
    print(Delay_from_due_date)
    
credit_history_age=str(int(Credit_Years))+" Years"+" and "+str(int(Credit_Months))+" Months "
credit_history_age=float(credit_history(credit_history_age))


    


button=st.button(label='Evaluate')
if button:
    test=(pd.DataFrame(data=[[Customer_ID,Month,Age,Occupation,Monthly_Inhand_Salary,
                 Num_Bank_Accounts,Num_Credit_Card,Num_of_Loan,
                 Delay_from_due_date,Changed_Credit_Limit,Outstanding_Debt,
                 credit_history_age,Payment_of_Min_Amount,Payment_Behaviour]],
                 columns=['Customer_ID', 'Month', 'Age', 'Occupation', 'Monthly_Inhand_Salary',
                'Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan',
                'Delay_from_due_date', 'Changed_Credit_Limit', 'Outstanding_Debt',
                'Credit_History_Age', 'Payment_of_Min_Amount', 'Payment_Behaviour']))
    print(test)
    y_pred=model.predict(test)
    if y_pred==1.0:
        st.write('Credit Score is Standard')
    elif y_pred==0.0:
        st.write('Credit Score is Poor')
    else:
        st.write('Credit Score is Good')
        