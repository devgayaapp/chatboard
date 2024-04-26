from pathlib import PosixPath
import boto3
import io
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SCRAPING_BUCKET
from botocore.exceptions import ClientError
from botocore.client import Config




def get_s3_client():
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(
            region_name = 'eu-west-2',
            signature_version='s3v4'
            )
    )
    return s3


def list_s3_keys(bucket_name):
    s3 = get_s3_client()
    objects = s3.list_objects_v2(Bucket=bucket_name)['Contents']
    sorted_objects = reversed(sorted(objects, key= lambda x: x['LastModified']))
    return [f.get('Key') for f in sorted_objects]
    

def get_s3_obj(bucket_name, key, read_first_line=False):
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    raw = io.BytesIO(obj['Body'].read())
    raw.seek(0)
    return raw


def upload_s3_obj(bucket_name, key, data):
    s3 = get_s3_client()
    return s3.upload_fileobj(data, bucket_name, key, ExtraArgs={'ACL': 'bucket-owner-full-control'})
    # s3.upload_fileobj(Bucket=bucket_name, Key=key, Body=data)


def upload_s3_file(filepath: PosixPath, bucket: str, filename: str=None):
    try:
        s3 = get_s3_client()
        filename = filename or filepath.name
        res = s3.upload_file(str(filepath), bucket, filename)
        return res
    except ClientError as e:
        print(e)
        raise Exception('client error uploading file to s3')



def delete_s3_obj(bucket_name, key):
    s3 = get_s3_client()
    return s3.delete_object(Bucket=bucket_name, Key=key)
    # return obj['Body'].read().decode('utf-8')
