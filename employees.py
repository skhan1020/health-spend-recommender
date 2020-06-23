import sqlite3
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt



def income(df_emp: pd.DataFrame) -> pd.DataFrame:
    """
    This function determines Income From Hourly Rate and Gross Income as well as Pay Type 

    Parameters
    ----------
    df_emp: pd.DataFrame
        pandas DataFrame with employee records

    Returns
    --------
    df2 : pd.DataFrame
        pandas DataFrame with employee id, annual income, and pay type
    """


    df = df_emp[['emp_id', 'pay_type',  'hourly_rate', 'gross_income']]

    df1 = df.set_index('emp_id')

    # Select Records which have null values in hourly_rate and gross_income
    df_null = df1[(df1['hourly_rate'].isnull() == True)  & (df1['gross_income'].isnull() == True)]
    
    df_noinc = deepcopy(df_null)
    
    # Assign null value to overall income
    df_noinc['income_from_hr'] = 'null'
    
    # Pay Type has a new category - Unknown
    df_noinc['pay_type'] = 'unknown'
    
    df_noinc1 = df_noinc[['pay_type', 'income_from_hr']]
    
    df_noinc2 = df_noinc1.reset_index()

    # Drop records whose hourly_rate and gross_income are both null. Replace remining null values with 0
    df2 = df1.dropna(how='all').fillna(0)

    # New Income Column : Hourly Rate * 2000 -- Average Annual Income    
    df2['income_from_hr'] = df2['hourly_rate']*2000

    # Yearly Income Calculated From Hourly Rate should correspond to Pay Type = 'hourly'
    df2['income_from_hr'] = np.where((df2['hourly_rate'] == 0) & (df2['pay_type'] == 'other'), df2['gross_income'], df2['income_from_hr'])

    # List Pay  Type  -  Hourly  with 0 Hourly Rates -- Need to be Removed
    df3 =  df2[(df2['pay_type'] == 'hourly') & (df2['hourly_rate'] == 0)]

    # Remove all records  with Hourly Rate = 0 and Pay Type = Hourly
    criteria1 = df2['pay_type'] == 'hourly'
    criteria2 = df2['hourly_rate'] == 0
    criteria = ~(criteria1 & criteria2)
    df2 = df2[criteria]


    # Remove all records with Gross Annual Income less  than 10,000
    df2 = df2[df2['income_from_hr'] > 5000]

    
    # Remove records whose (gross_income/ income_fro_hr) ratio deviates significantly 
    df2['ratio'] = df2['gross_income']/df2['income_from_hr']
    df2 = df2[(df2['ratio'] ==  0) | ((df2['ratio'] >=0.8) & (df2['ratio'] <= 1.2))]

    df2 = df2[['pay_type', 'income_from_hr']]

    df3 = df2.reset_index()
    
    # Concatenate income dataframes with Null and Non-Null values
    df4 = pd.concat([df3, df_noinc2])

    return df4




def gender(df_emp: pd.DataFrame) -> pd.DataFrame:
    """
    This function performs Gender and Gender Confidence Analysis From Employees Table

    Parameters
    ----------
    df_emp: pd.DataFrame
        pandas DataFrame with employee records

    Returns
    --------
    df_copy : pd.DataFrame
        pandas DataFrame with employee id, gender, gender confidence
    """
    df = df_emp[['emp_id', 'gender', 'gender_confidence']]

    df_copy = deepcopy(df)
    df_copy['gender'].fillna('unknown', inplace=True)
    df_copy['gender_confidence'] = np.where(df_copy['gender'] == 'unknown', 0.5, df_copy['gender_confidence'])

    return df_copy



def filing(df_emp: pd.DataFrame) -> pd.DataFrame:
    """
    This function determines Filing Status of Employees

    Parameters
    ----------
    df_emp: pd.DataFrame
        pandas DataFrame with employee records

    Returns
    --------
    df1 : pd.DataFrame
        pandas DataFrame with employee id, filing status
    """

    df  = df_emp[['emp_id', 'filing_status']]
    df1 = deepcopy(df)
    df1['filing_status'].fillna('no_status', inplace=True)

    return df1




def age(df_emp: pd.DataFrame) -> pd.DataFrame:
    """
    This function determines Age Category of Employees

    Parameters
    ----------
    df_emp: pd.DataFrame
        pandas DataFrame with employee records

    Returns
    --------
    df4 : pd.DataFrame
        pandas DataFrame with employee id, age, age category
    """

    df = df_emp[['emp_id', 'birth_year']]
    df1 = deepcopy(df)

    # For  Employees with  null value in birth_year, age_cateogry = no_age
    df1['birth_year'].fillna('no_age', inplace=True)

    
    df2 = df1[df1['birth_year'] != 'no_age']

    df2_copy = deepcopy(df2)

    # Calculate Age of Employees
    df2_copy['age'] = 2020 - np.int64(df2_copy['birth_year'])

    df3 = df2_copy.drop(columns=['birth_year'])

    # Remove employees whose ages have been entered incorrectly
    df3 =  df3[(df3['age'] > 10) & (df3['age'] < 100)]

    # Categorise employees into four different Age Categories : ''
    df3['age_category'] =  pd.cut(df3['age'], bins=[10, 30, 50, 70, 100], labels=['young_adult', 'adult', 'middle_aged', 'retired'])

    # People belonging 'no_age' category are given a default value of 0
    df1 = df1[df1['birth_year'] == 'no_age']
    df1['age'] = 0
    df1.rename({'birth_year':'age_category'}, axis=1, inplace=True)

    df4 = pd.concat([df1, df3])

    return df4




def city(df_emp: pd.DataFrame):
    """
    This function returns the city id of Employees

    Parameters
    ----------
    df_emp: pd.DataFrame
        pandas DataFrame with employee records

    Returns
    --------
    df: pd.DataFrame
      pandas DataFrame with employee id, city id
    """

    df =  df_emp[['emp_id', 'city_id']]
    df['city_id'].fillna(0, inplace=True)

    return  df



def result(db: str) -> pd.DataFrame:
    """
    This function merges dataframes of employees containing (income, pay type), (gender,
    gender confidence), (age category, filing status), and city id from
    Employees  Table

    Parameters
    ----------
    db: str
      name of sqlite database

    Returns
    --------
    df3 : pd.DataFrame
        pandas DataFrame with employee id, income, pay type, gender, gender
        confidence, age category, filing status, ctiy id
    """

    conn = sqlite3.connect(db)

    df = pd.read_sql_query("Select * From Employees", conn)

    # Rename Hashed Employee Id as Emp Id
    df.rename({'hashed_employee_id':'emp_id'}, axis=1, inplace=True)

    # Convert Income and Hourly rate to dollars
    df['gross_income'] = np.float64(df['gross_income']/100)
    df['hourly_rate'] = np.float64(df['hourly_rate']/100)

    # Removing language preference since only one category is included
    df.drop({'language_preference'}, axis=1, inplace=True)
    
    df_inc = income(df)
    df_gen = gender(df)
    df_fil = filing(df)
    df_age = age(df)
    df_city = city(df)

    # Merge Income and Gender DataFrames
    df1 = df_inc.merge(df_gen, on='emp_id', how='inner')

    # Merge Filing DataFrame
    df2 = df1.merge(df_fil, on='emp_id', how='inner')

    # Merge Age DataFrame
    df3 = df2.merge(df_age, on='emp_id', how='inner')

    # Merge City ID DataFrame
    df4 = df3.merge(df_city, on='emp_id', how='inner')

    
    return df4
