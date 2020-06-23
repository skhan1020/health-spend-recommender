from glob  import glob
import pandas as pd
import sqlite3


def transactions_table(filepath: str,  engine: callable) -> None:
    """
    Creates Transactions table inside a sqlite database after
    reading csv files from ./data/txns/

    Parameters
    ----------
    fileapth: str
        path of file excluding filename
    engine: callable
        Engine instance to connect to sqlite database
    """
    dfs = list()

    filenames = glob(filepath +  '*.csv')
    for filename in  filenames:
        df =  pd.read_csv(filename, encoding='latin-1', low_memory=False)
        dfs.append(df)

    big_frame = pd.concat(dfs, ignore_index=True, sort=True)

    big_frame.to_sql('Transactions', con=engine, index=False, if_exists='append')

def employees_table(filepath: str, engine: callable) -> None:
    """
    Creates Employees table inside a sqlite database after
    reading csv file employees_data.csv from ./data/emp/

    Parameters
    ----------
    fileapth: str
        path of file including filename
    engine: callable
        Engine instance to connect to sqlite database
    """

    df = pd.read_csv(filepath)
    df.to_sql('Employees', con=engine, index=False, if_exists='append')
