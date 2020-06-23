import boto3


class S3Client:
    """ Object to wrap boto3 s3 services
        http://boto3.readthedocs.io/en/latest/reference/services/s3.html#S3.Client
    """

    def __init__(self):
        self.conn = boto3.client("s3")

    def get_object(self, **kwargs):
        return self.conn.get_object(**kwargs)

    def put_object(self, **kwargs):
        return self.conn.put_object(**kwargs)

    def copy_object(self, **kwargs):
        return self.conn.copy_object(**kwargs)

    def download_file(self, bucket, key, filename, **kwargs) -> None:
        """ Downloads file from bucket to local machine

            Parameters
            ----------
            bucket: str
                name of s3 bucket
            key: str
                key file is located at
            filename: str
                filename (and path) to download file to on local machine
            """
        self.conn.download_file(bucket, key, filename, **kwargs)

    def upload_file(self, filename, bucket, key, **kwargs) -> None:
        """ Uploads file from local machine to bucket

            Note the different order of arguments (vs. download_file)

            Parameters
            ----------
            filename: str
                filename (and path) of file on local machine
            bucket: str
                name of s3 bucket to put file in
            key: str
                key file is will be located at
        """
        self.conn.upload_file(filename, bucket, key, **kwargs)
