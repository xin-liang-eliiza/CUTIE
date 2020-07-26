import os
import json
from typing import Dict

from PIL import Image
from io import BytesIO
import numpy as np
from IPython.display import display

import boto3


s3 = boto3.resource('s3')


def display_image_from_s3(bucket: str, key: str):
    obj = s3.Bucket(bucket).Object(key)
    im = Image.open(obj.get()['Body'])
    display(im)


def read_json_from_s3(bucket: str, key: str) -> Dict:
    obj = s3.Bucket(bucket).Object(key)
    json_data = json.loads(obj.get()['Body'].read().decode('utf-8'))
    return json_data


