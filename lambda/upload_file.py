import json
import os
import urllib.parse

import requests

VALID_FILE_TYPES = (".pdf", ".txt", ".md")
TOKEN_ENDPOINT = f"{os.getenv('API_URL')}/token"
UPLOAD_ENDPOINT = f"{os.getenv('API_URL')}/upload"
CDN_URL = f"{os.getenv('CDN_URL')}"


def lambda_handler(event, context):
    try:
        record = event["Records"][0]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])

        if not key.lower().endswith(VALID_FILE_TYPES):
            return {"statusCode": 200, "body": f"Skipped {key}"}

        username = os.getenv("USERNAME")
        password = os.getenv("PASSWORD")
        if not username or not password:
            return {
                "statusCode": 500,
                "body": json.dumps({"error": "Missing Username or Password"}),
            }

        token_response = requests.post(
            TOKEN_ENDPOINT,
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )

        if token_response.status_code != 200:
            return {
                "statusCode": token_response.status_code,
                "body": json.dumps(
                    {"error": "Failed to obtain token", "details": token_response.text}
                ),
            }

        token = token_response.json()["access_token"]

        if not token:
            return {"statusCode": 500, "body": "Missing token"}

        payload = {"url": f"{CDN_URL}/{key}"}

        upload_response = requests.post(
            UPLOAD_ENDPOINT,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=10,
        )

        return {"statusCode": upload_response.status_code, "body": upload_response.text}

    except Exception as e:
        print(f"Error: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
