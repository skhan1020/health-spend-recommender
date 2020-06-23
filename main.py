from s3 import S3Client
from s3data import get_transaction_data, get_employee_data
from sqlalchemy  import create_engine
from makedb import transactions_table, employees_table
from emptransac import employee_transac
from model import RandomForest

if __name__ == '__main__':

    s3 = S3Client()
    bucket =  'alice-transaction-data'
    key = 'fsa-recommender-data'

    get_transaction_data(s3, bucket, key)
    get_employee_data(s3, bucket, key)

    db = 'fsa.db'
    engine = create_engine('sqlite:///' + db, echo=False)

    filepath  = './data/txns/'
    transactions_table(filepath, engine)

    filepath = './data/emp/employees_data.csv'
    employees_table(filepath, engine)

    nclasses = 3
    breakpoints = [750, 1500]
    income_threshold = 15000
    amount_threshold = 100

    employee_transac(db, nclasses, breakpoints, amount_threshold,\
            income_threshold, engine)


    answer = input('ROC Curve or Confusion Matrix ?  ')

    if answer == 'Confusion Matrix':
        RandomForest(db, cf_matrix=True)
    elif answer == 'ROC Curve':
        RandomForest(db, cf_matrix=False)
    else:
        print('Invalid Option Chosen!')

