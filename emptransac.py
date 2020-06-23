import sqlite3
import sqlalchemy
import pandas as pd
import numpy as np
from dask import delayed
import dask.dataframe as dd
from inputfeatures import attributes, classify_income
from target import amount


def merge_dataframe(db: str, nclasses: int, intervals: list, amt_thresh: float, inc_thresh: float) -> pd.DataFrame:
    """
    This function generates the combined pandas DataFrame with both employee
    attributes  and their respective transaction  history

    Parameters
    ----------
    db:  str
        name of sqlite database
    nclasses:  int
        number of target variable classes
    intervals:  list
        endpoints of amount groups
    amt_thresh:  float
        threshold value to merge two amount groups
    inc_thresh:  float
        threshold value to merge two income groups
    Returns
    --------
    df_merged2:  pd.DataFrame
        merged pandas DataFrame with employee attributes and target variable
    """
    conn = sqlite3.connect(db)

    df_emp =  attributes(db)
    df_tx = amount(db, nclasses, intervals, amt_thresh)

    df_tx1 = df_tx.set_index('emp_id')
    df_emp1  = df_emp.set_index('emp_id')

    partitions =  []
    for i in  range(10):
        partitions.append(delayed(df_tx1))

    dd_tx = dd.from_delayed(partitions,  df_tx1)
    
    df_merged =  dd.merge(df_tx1, df_emp1)

    df_merged1 =  df_merged.reset_index()
    
    df_merged2 = classify_income(df_merged1, inc_thresh)
    
    conn.commit()

    return df_merged2

def employee_transac(db: str, nclasses: int, intervals: list, amt_thresh: float, inc_thresh: float, engine: callable) -> None:
    """
    This function creates the  EmployeesTransactions Table in the sqlite
    database

    Parameters
    ----------
    db:  str
        name of sqlite database
    nclasses:  int
        number of target variable classes
    intervals:  list
        endpoints of amount groups
    amt_thresh:  float
        threshold value to merge two amount groups
    inc_thresh:  float
        threshold value to merge two income groups
    engine:  callable
        Engine instance to connect to sqlite database
    """

    df = merge_dataframe(db, nclasses, intervals, amt_thresh, inc_thresh)
    df.to_sql('EmployeesTransactions', con=engine, index=False, if_exists='append')

