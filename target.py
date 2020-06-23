import sqlite3
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns

def spend(db: str) -> pd.DataFrame:
    """
    This function generates calculates amount of all employees from
    Transactions table who spent their money on  healthcare, dental
    or vision variable,  using a sliding  window  approach

    Parameters
    ----------
    db:   str
        name of sqlite database

    Returns
    -------
    df8:   pd.DataFrame
        pandas DataFrame with employee id and the average yearly expenditure on
        healthcare
    """

    conn = sqlite3.connect(db)

    df_et = pd.read_sql_query('Select hashed_owner_id, pretax_category_truth_label, amount, date From Transactions', conn)
    df_et.rename({'hashed_owner_id':'emp_id', 'pretax_category_truth_label':'category'}, axis=1, inplace=True)

    # Convert cents to dollars
    df_et['amount'] = np.float64(df_et['amount']/100)

    # Filter Healthcare Related Expenses
    condition = (df_et['category'] == 'healthcare') | (df_et['category'] == 'vision') | (df_et['category'] == 'dental')
    df_et = df_et[condition]

    df_et['date'] = pd.to_datetime(df_et['date'])
    df_et['month_year'] = df_et['date'].dt.to_period('M')

    df1 = df_et[['emp_id', 'month_year', 'amount']]

    # Aggregate monthly expenses
    df2 = df1.groupby(['emp_id', 'month_year']).agg({'amount':'sum'}).reset_index()

    df3 = deepcopy(df2)

    # Create Sliding Windows For Yearly Transactions
    df3['start_date'] = df3['month_year'].dt.strftime('%Y-%m').add('-01')
    df3['start_date'] = pd.to_datetime(df3['start_date'])
    df3['end_date'] = df3['start_date'] + pd.DateOffset(years=1)
    df3['date'] = df3['month_year'].dt.strftime('%Y-%m').add('-01')
    df3['date'] = pd.to_datetime(df3['date'])

    # Determine Last Date of Transaction for Each Employee
    df4 = df3[['emp_id', 'date']]

    df5 = deepcopy(df4)

    df5['max_date'] =  df5.groupby('emp_id')['date'].transform(max)
    df5.drop(columns=['date'], inplace=True)

    df6 = df3.merge(df5, on='emp_id', how='inner')
    df7  = df6.drop_duplicates()

    def groupby_rolling(grp_df):


        df1 = grp_df.apply(lambda x : grp_df.loc[(grp_df.date >= x.start_date)
            & (grp_df.date <=x.end_date), 'amount'].sum(), axis=1)
        df2 = grp_df.apply(lambda x: grp_df.loc[(grp_df.date >= x.start_date) &
                                                (grp_df.date <= x.end_date), 'amount'].count(), axis=1)

        df3 = pd.DataFrame({'amountSum':df1, 'periodCount':df2, 'emp_id':grp_df['emp_id']})
        df3['avg_amount'] = 12*df3['amountSum']/df3['periodCount']
        df4 = df3.groupby('emp_id').agg({'avg_amount':'mean'}).reset_index()


        return df4

    df8 = df7.groupby("emp_id").apply(groupby_rolling).drop(columns=['emp_id']).reset_index().drop(columns=['level_1'])

    # Retain Transactions that are Non Negative and Less than $3,000 (well
    # above the  maximum FSA)
    df8 = df8[(df8['avg_amount'] >= 0)  & (df8['avg_amount'] < 3000)]


    return df8



def nospend(db: str) -> pd.DataFrame:
    """
    This function selects all those employees from
    Transactions table who did not spend any money on  healthcare, dental
    or vision variable and  have 12 months of transactions recorded in a year 

    Parameters
    ----------
    db:   str
        name of sqlite database

    Returns
    -------
    df3:   pd.DataFrame
        pandas DataFrame with employee id and the average amount (zero spend)
    """

    conn = sqlite3.connect(db)

    df_et = pd.read_sql_query('Select hashed_owner_id, pretax_category_truth_label, amount, date From Transactions', conn)
    df_et.rename({'hashed_owner_id':'emp_id', 'pretax_category_truth_label':'category'}, axis=1, inplace=True)


    # Filter Non Healthcare Related Expenses
    condition = (df_et['category'] == 'healthcare') | (df_et['category'] == 'vision') | (df_et['category'] == 'dental')
    df_et = df_et[~condition]

    df_et['date'] = pd.to_datetime(df_et['date'])
    df_et['year'] = pd.DatetimeIndex(df_et['date']).year
    df_et['month'] = pd.DatetimeIndex(df_et['date']).month

    df1 = df_et[['emp_id', 'month', 'year']]
    df2 = df1.drop_duplicates()

    # Count the number of monthly transactions
    df3 = df2.groupby(['emp_id', 'year']).agg(transactions=('month', 'count')).reset_index()

    # Retain employees with at least 12  months of transaction data
    df3 =  df3[df3['transactions'] == 12]
    df3['amount'] = 0.0
    df3.drop(columns=['transactions', 'year'], inplace=True)

    return df3



def amount(db: str, nclasses: int, intervals: list, amount_threshold: float, add_nospend: bool = True) -> pd.DataFrame:
    """
    This function generates the target variable -- average yearly expenditure
    on healthcare from EmployeesTransactions table

    Parameters
    ----------
    db:   str
        name of sqlite database
    nclasses:  int
        number of classes/groups of amount variable
    intervals:  list
        endpoints of the interval ranges that determine the groups, excludes
        first and last endpoints
    amount_threshold:  float
        threshold value to merge different amount groups/classes
    add_nospend:  bool
        boolean variable to decide whether to employees with zero spend

    Returns
    -------
        pandas DataFrame with employee id and the amount class (target
        variable)
    """

    df_spend = spend(db)
    df = df_spend[df_spend['avg_amount'] > 0]

    # Group Transactions into Different Categories
    max_amt =  df['avg_amount'].describe()['max']
    intervals.insert(1, 0)
    intervals.append(max_amt)

    df1 = deepcopy(df)

    df1['amount_range'] = pd.cut(df1['avg_amount'], bins=sorted(intervals))
    df1['target'] = df1['amount_range'].apply(lambda x : x.mid)

    df1.drop(columns=['avg_amount', 'amount_range'], inplace=True)
    df1.rename({'target':'amount'}, axis=1, inplace=True)

    df1['amount'] = np.float64(df1['amount'])

    amt_vals = sorted(df1['amount'].unique())

    diff = np.diff(amt_vals)

    threshold = amount_threshold

    for i, num in enumerate(diff):
        if num < threshold:
            df1['amount'] = np.where(df1['amount'] == amt_vals[i], amt_vals[i+1], df1['amount'])
            amt_vals.remove(amt_vals[i])

    class_labels = list(range(nclasses))
    df1['amount'] = df1['amount'].map(dict(zip(amt_vals, class_labels)))

    if add_nospend ==  True:

        df_nospend = nospend(db)
        df_spend0 = df_spend[df_spend['avg_amount'] == 0.0]

        df2 = deepcopy(df_nospend)
        df3 = deepcopy(df_spend0)
        df3.rename({'avg_amount':'amount'}, axis=1, inplace=True)

        df4 = pd.concat([df2,  df3])
        df4['amount'] = min(class_labels)

        df5 =  pd.concat([df1,  df4])

        return df5

    else:

        return df1
