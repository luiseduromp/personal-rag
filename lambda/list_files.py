import json
import os

import boto3
from botocore.exceptions import ClientError

s3_client = boto3.client("s3")


def lambda_handler(event, context):
    """
    Lambda handler to list all files in the /docs folder of an S3 bucket.

    The bucket name is retrieved from the S3_BUCKET_NAME environment variable.
    It returns a JSON response suitable for API Gateway Lambda proxy integration.
    """
    bucket_name = os.environ.get("S3_BUCKET_NAME")
    if not bucket_name:
        print("Error: S3_BUCKET_NAME environment variable not set.")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(
                {"error": "S3_BUCKET_NAME environment variable not set."}
            ),
        }

    prefix = "docs/"
    files = []

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        for page in pages:
            if "Contents" in page:
                for item in page["Contents"]:
                    if not item["Key"].endswith("/"):
                        files.append(item["Key"])

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"files": files}),
        }

    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchBucket":
            print(f"Error: Bucket '{bucket_name}' not found.")
            return {
                "statusCode": 404,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": f"Bucket '{bucket_name}' not found."}),
            }
        print(f"An unexpected Boto3 client error occurred: {e}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Failed to list files from S3."}),
        }

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "An unexpected error occurred."}),
        }
