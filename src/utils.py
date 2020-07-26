import os
import json
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import numpy as np
from IPython.display import display

import boto3


s3 = boto3.resource('s3')


def display_image_from_s3(bucket: str, key: str, bboxes: List = [], displayed=False):
    font = ImageFont.truetype("./Hack-v3.003-ttf/ttf/Hack-Regular.ttf", 25)

    obj = s3.Bucket(bucket).Object(key)
    im = Image.open(obj.get()['Body'])
    if bboxes:
        draw = ImageDraw.Draw(im)
        for ind, box in enumerate(bboxes):
            draw.rectangle(box, outline="#b283d4", width=3)
            draw.text((box[0]-5, box[3]), str(ind), font=font, fill="#FF0000")
    if displayed:
        display(im)
    return im


def read_json_from_s3(bucket: str, key: str) -> Dict:
    obj = s3.Bucket(bucket).Object(key)
    json_data = json.loads(obj.get()['Body'].read().decode('utf-8'))
    return json_data


