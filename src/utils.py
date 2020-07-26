import os
import json
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import numpy as np
from IPython.display import display

import boto3


s3 = boto3.resource('s3')


def display_image_from_s3(bucket: str, key: str, boxes: List = [], displayed=False):
    font = ImageFont.truetype("./Hack-v3.003-ttf/ttf/Hack-Regular.ttf", 20)

    obj = s3.Bucket(bucket).Object(key)
    im = Image.open(obj.get()['Body'])
    if boxes:
        draw = ImageDraw.Draw(im)
        for b in boxes:
            draw.rectangle(b.bbox, outline="#b283d4", width=3)
            draw.text((b.bbox[0]-5, b.bbox[3]), str(b.id), font=font, fill="#FF0000")
    if displayed:
        display(im)
    return im


def read_json_from_s3(bucket: str, key: str) -> Dict:
    obj = s3.Bucket(bucket).Object(key)
    json_data = json.loads(obj.get()['Body'].read().decode('utf-8'))
    return json_data


def write_json_to_s3(json_data, bucket: str, key: str):
    obj = s3.Bucket(bucket).Object(key)
    obj.put(Body=json.dumps(json_data, indent=4), ServerSideEncryption="AES256")


def write_image_to_s3(pil_img, bucket: str, key: str):
     out_img = BytesIO()
     pil_img.save(out_img, format="png")
     out_img.seek(0)
     s3.Bucket(bucket).put_object(
         Key=key,
         Body=out_img,
         ContentType="image/png",
         ServerSideEncryption="AES256",
     )


def list_s3_keys(bucket: str, prefix: str) -> List:
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = [i["Key"] for i in response.get("Contents", [])]
    return keys

