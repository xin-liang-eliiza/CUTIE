import os
import json
from typing import Dict, List
#from fuzzywuzzy import fuzz
import dateparser

from PIL import Image, ImageDraw, ImageFont
from io import BytesIO, StringIO
import numpy as np
import pandas as pd
from IPython.display import display
from shapely.geometry import box as polygon

import boto3


s3 = boto3.resource('s3')


def read_image_from_s3(bucket: str, key: str, boxes: List = [], displayed=False, to_file=True):
    font = ImageFont.truetype("./Hack-v3.003-ttf/ttf/Hack-Regular.ttf", 20)

    obj = s3.Bucket(bucket).Object(key)
    im = Image.open(obj.get()['Body'])
    im_file = key.rpartition('/')[-1]
    im.save("../eval_data/{}".format(im_file))
    if boxes:
        draw = ImageDraw.Draw(im)
        for b in boxes:
            draw.rectangle(b.bbox, outline="#b283d4", width=3)
            draw.text((b.bbox[0]-5, b.bbox[3]), str(b.id), font=font, fill="#FF0000")
    if displayed:
        display(im)
    return im


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


def read_json_from_s3(bucket: str, key: str) -> Dict:
    obj = s3.Bucket(bucket).Object(key)
    json_data = json.loads(obj.get()['Body'].read().decode('utf-8'))
    return json_data


def write_json_to_s3(json_data, bucket: str, key: str):
    obj = s3.Bucket(bucket).Object(key)
    obj.put(Body=json.dumps(json_data, indent=4), ServerSideEncryption="AES256")


def download_json_from_s3(bucket: str, keys: List, out_dir: str):
    for key in keys:
        file_name = key.rpartition('/')[-1]
        json_data = read_json_from_s3(bucket, key)
        if json_data['fields']:
            json.dump(json_data, open("{}/{}".format(out_dir, file_name), "w"), indent=4)
    


def read_csv_from_s3(bucket: str, key: str):
    obj = s3.Bucket(bucket).Object(key)
    csv = obj.get()['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(csv))
    return df


def list_s3_keys(bucket: str, prefix: str) -> List:
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = [i["Key"] for i in response.get("Contents", [])]
    return keys


#def is_text_matched(text_1, text_2):
#    r = fuzz.ratio(text_1, text_2)
#    return (r == 100)


def is_date_matched(dt_str_1, dt_str_2):
    is_matched = False

    def parse_dt(dt_str):
        if '/' in dt_str:
            dt = dateparser.parse(dt_str, settings={'DATE_ORDER': 'DMY'})
        else:
            dt = dateparser.parse(dt_str)
        return dt

    dt_1 = parse_dt(dt_str_1)
    dt_2 = parse_dt(dt_str_2)
    if dt_1 is not None and dt_2 is not None and dt_1 == dt_2:
        is_matched = True
    return is_matched


def get_overlap_bbox(pred_bbox, text_boxes, overlap_thresh=0.9):
    pred_poly = polygon(pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3])
    for item in text_boxes:
        bbox = item['bbox']
        tpoly = polygon(bbox[0], bbox[1], bbox[2], bbox[3])
        overlap = pred_poly.intersection(tpoly)
        if (overlap.area / pred_poly.area) > overlap_thresh:
            return item
    return None
