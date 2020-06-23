import sqlite3
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from employees import  result

def attributes(db: str) ->  pd.DataFrame:
    """
    This function builds the input features from  Employees
    Table

    Parameters
    ----------
    db:   str
        name of sqlite database

    Returns
    --------
    df1:  pd.DataFrame 
        pandas DataFrame with input features
    """

    df_et  = result(db)

    df_et1 = df_et[df_et['income_from_hr'] != 'null']
    df = deepcopy(df_et1)

    # Modifying Income 

    df.rename({'income_from_hr':'income'}, axis=1, inplace=True)

    # Ignore employees with income greater than $250,000
    df = df[df['income'] < 2.5*10**5]


    # Employee with Null Income

    df_et2 = df_et[df_et['income_from_hr'] == 'null']
    df1 = deepcopy(df_et2)
    df1.rename({'income_from_hr':'income'}, axis=1, inplace=True)
    df1['income'] =  'null'

    # Final DataFrame
    df2 = pd.concat([df, df1])

    # One Hot Encoding of 'gender' and combining the result with the respective 'gender_confidence' value

    df3 = pd.get_dummies(df2, columns=['gender'])
    df3['gender_female'] = np.where(df3['gender_female'] ==1, df3['gender_confidence'], 0)
    df3['gender_male'] = np.where(df3['gender_male'] ==1, df3['gender_confidence'], 0)
    df3['gender_unknown'] = np.where(df3['gender_unknown'] ==1, df3['gender_confidence'], 0)

    # Drop 'gender_confidence', 'pay_type' (since it is correlated with income), and 'age'
    df3.drop(columns=['gender_confidence', 'pay_type', 'age'], inplace=True)

    # One Hot Encoding of other features
    df3 = pd.get_dummies(df3, columns=['filing_status', 'age_category'])

    return df3

def classify_income(df: pd.DataFrame, income_threshold:  float, inc_null: bool = True) -> pd.DataFrame:
    """
    This function converts income to categorical variable

    Parameters
    ----------
    income_threshold:   float
        threshold for merging two income groups
    inc_null:   bool
        boolean to include 'null' values in income

    Returns
    --------
    df:  pd.DataFrame 
        pandas DataFrame with income groups
    """

    if  inc_null:
        df1 = df[df['income'] == 'null']
        df  = df[df['income'] != 'null']  

    # Convert str to float
    df['income'] = np.float64(df['income'])

    # Generate Income Quartiles

    amt_25 = df['income'].describe()['25%']
    amt_50 = df['income'].describe()['50%']
    amt_75 = df['income'].describe()['75%']
    max_amt =  df['income'].describe()['max']


    df['income_range'] = pd.cut(df['income'], bins=[0, amt_25, amt_50, max_amt])
    df['target'] = df['income_range'].apply(lambda x : x.mid)

    df.drop(columns=['income', 'income_range'], inplace=True)
    df.rename({'target':'income'}, axis=1, inplace=True)

    df['income'] = np.float64(df['income'])

    income_vals = sorted(df['income'].unique())
    diff = np.diff(income_vals)
    threshold = income_threshold
    for i, num in enumerate(diff):
        if num < threshold:
            df['income'] = np.where(df['income'] == income_vals[i],\
                    income_vals[i+1], df['income'])
            income_vals.remove(income_vals[i])

    # Income Class Labels

    if len(income_vals) == 2:
        class_labels = ['low', 'high']
    else:  
        class_labels =  ['low', 'medium', 'high'] 

    df['income'] = df['income'].map(dict(zip(income_vals, class_labels)))

    
    if  inc_null:
        
        # Merge DataFrames with 'null' values
        df2 =  pd.concat([df, df1])

        # One Hot Encoding of Income
        df2 = pd.get_dummies(df2, columns=['income'])

        return df2
    else:

        # One Hot Encoding of Income
        df = pd.get_dummies(df, columns=['income'])
        return  df
        
