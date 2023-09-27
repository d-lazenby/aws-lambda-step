import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    key = event["s3_key"]
    bucket = event["s3_bucket"]

    img_loc = f"/tmp/{key[5:]}" 
    
    s3.download_file(bucket, key, img_loc)

    with open(img_loc, "rb") as f:
        image_data = base64.b64encode(f.read())

    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }
