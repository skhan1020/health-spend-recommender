from s3 import S3Client
from botocore.exceptions import ClientError
from datetime import  timedelta, date


def daterange(start_date: date, end_date: date):
    """
    This function returns a generator to produce a list of all possible days
    within a certain time period

    Parameters
    ----------
    start_date:  date
        starting date of time period
    end_date:  date
        end date of time period
    """

    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def get_transaction_data(s3_obj: S3Client, bucket_name: str, key_name: str):
    """
    This function downloads files pertaining to transaction data from an s3
    bucket

    Parameters
    ----------
    s3_obj:  S3Client
        S3Client instance that wraps boto3 s3 services
    bucket_name:  str
        name of s3 bucket
    key_name: str
        name of s3 key
    """

    start_date = date(2017, 10, 22)
    end_date = date(2020, 4, 22)

    for single_date in daterange(start_date, end_date):

        key = key_name + single_date.strftime("%Y-%m-%d") +'/txns.csv'
        filepath = './data/txns/txns' + single_date.strftime("%Y-%m-%d") + '.csv'
        try:
            s3_obj.download_file(bucket_name, key, filepath)
        except ClientError as e:

            print( key + 'Not Found!')

            if e.response['Error']['Code'] == 404:
                pass

def get_employee_data(s3_obj: S3Client, bucket_name: str, key_name:  str):
    """
    This function downloads files containing personal attributes of employees from an s3
    bucket

    Parameters
    ----------
    s3_obj:  S3Client
        S3Client instance that wraps boto3 s3 services
    bucket_name:  str
        name of s3 bucket
    key_name:  str
        name of s3 key
    """

    key = key_name + 'employees_data.csv'
    filepath = './data/emp/employees_data.csv'
    
    try:
        s3_obj.download_file(bucket_name, key, filepath)
    except ClientError as e:

        print( key + 'Not Found!')

        if e.response['Error']['Code'] == 404:
            pass
