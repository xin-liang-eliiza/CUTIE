import pandas as pd
import math
import json
from typing import List

from utils import read_image_from_s3, read_json_from_s3, write_json_to_s3, write_image_to_s3, list_s3_keys, read_csv_from_s3, is_date_matched, is_text_matched
from format_textract import get_textract_blocks_by_type
from format_data import format_text_boxes, get_global_attribute, generate_input_data, label_classes, get_field_label


#file_id = "0318922720232952_0.jpg"

landing_bucket = "nib-prod-analytics-privacy-landing-ap-southeast-2"
im_key_prefix = "claim-receipt-recognition/category_id_19/{}"

transform_bucket = "nib-prod-analytics-privacy-transformed-ap-southeast-2"
textract_key_prefix = "claim-receipt-recognition/textract_raw_results/2020/01/category_id_19/{}_textract.json"
anno_text_key_prefix = "claim-receipt-recognition/annotations/text/category_id_19/{}_annotation.json"
anno_im_key_prefix = "claim-receipt-recognition/annotations/images/category_id_19/{}"
label_text_key_prefix = "claim-receipt-recognition/annotations/labels/category_id_19/{}_label.json"
auto_label_text_key_prefix = "claim-receipt-recognition/annotations/auto_labels/category_id_19/{}_auto_label.json"

cas_csv_key = "claim-receipt-recognition/CAS submissions 2019-11-01 to 2020-01-12.csv"


def annotate_data(file_ids):
    for fid in file_ids:
        try:
            print("Annotate data for file id {}".format(fid))
            # Generate annotated text data
            t_key = textract_key_prefix.format(fid)
            json_data = read_json_from_s3(transform_bucket, t_key)
            blocks = get_textract_blocks_by_type(json_data)

            i_key = im_key_prefix.format(fid)
            im = read_image_from_s3(landing_bucket, i_key, displayed=False)
            text_boxes = format_text_boxes(blocks, im.width, im.height)
            global_attr = get_global_attribute(fid)

            input_data = generate_input_data([t._asdict() for t in text_boxes], [], global_attr._asdict())
            write_json_to_s3(input_data._asdict(), transform_bucket, anno_text_key_prefix.format(fid))

            # Generate annotated image data
            im = read_image_from_s3(landing_bucket, i_key, text_boxes, displayed=False)
            write_image_to_s3(im, transform_bucket, anno_im_key_prefix.format(fid))
        except Exception as e:
            print("Error occurred for file id {}: {}".format(fid, repr(e)))


def format_labeled_data(label_csv):
    df = pd.read_csv(label_csv)
    #df.drop(columns=['comments'])
    for index, row in df.iterrows():
        if isinstance(row['comments'], str):
            continue
        label_fields = []
        file_id = row['file_id']
        file_key = anno_text_key_prefix.format(file_id)
        anno_json = read_json_from_s3(transform_bucket, file_key)
        for cls in label_classes:
            if isinstance(row[cls], str):
                value_ids = list(map(int, str(row[cls]).split(',')))
            elif not math.isnan(row[cls]):
                value_ids = [int(row[cls])]
            else:
                value_ids = []
            value_text = [t["text"] for t in anno_json['text_boxes'] if t['id'] in value_ids]
            field_label = get_field_label(cls, value_ids, value_text)
            label_fields.append(field_label._asdict())
        anno_json['fields'] = label_fields
        print("key", label_text_key_prefix.format(file_id))
        print(label_fields)
        write_json_to_s3(anno_json, transform_bucket, label_text_key_prefix.format(file_id))
        json.dump(anno_json, open("../invoice_data/{}.json".format(file_id), 'w'), indent=4)
    


def gt_class_mapping():
    mapping = {
        "provider_number.1": "ProviderNum",
        "claim_item_number": "ItemNum",
        "tooth_code": "ToothId",
        "date_of_service": "ServiceDate",
        "item_fee": "ItemCharge"
    }
    return mapping

            
def label_gt(anno_bucket: str = transform_bucket, gt_bucket: str = landing_bucket, cas_csv_key: str = cas_csv_key, anno_text_key_prefix: str = anno_text_key_prefix):
    '''
    Programmatically label data using histocial ground truth from CAS
    '''
    columns = ['photo_tracking_number', 'category_id', 'member_number', 'page_number', 'provider_number.1', 'claim_item_number', 'claim_item_entry', 'tooth_code', 'date_of_service', 'item_fee', 'script_number']
    gt_columns = ['provider_number.1', 'claim_item_number', 'tooth_code', 'date_of_service', 'item_fee']
    
    anno_text_list = list_s3_keys(anno_bucket, anno_text_key_prefix.rpartition('/')[0])
    gt_df = read_csv_from_s3(gt_bucket, cas_csv_key)
    gt_df = gt_df[columns]
    gt_df = gt_df[gt_df['photo_tracking_number'].notna()]
    gt_df["photo_tracking_number"] = gt_df["photo_tracking_number"].apply(int).apply(str)
    for key in anno_text_list:
        file_name = key.rpartition('/')[-1]
        tracking_num, suffix = file_name.split('_')
        page_num, _ = suffix.split('_')
        tracking_num = tracking_num.lstrip('0')

        anno_json = read_json_from_s3(anno_bucket, key)
        print("tracking num", tracking_num)
        gt = gt_df[gt_df["photo_tracking_number"] == tracking_num] 
        gt = gt.sort_values('page_number')
        text_boxes = anno_json['text_boxes']
        label_fields = match_text(text_boxes, gt, gt_columns)
        anno_json['fields'] = label_fields

        print("key", auto_label_text_key_prefix.format(file_name.split('.')[0]))
        write_json_to_s3(anno_json, transform_bucket, auto_label_text_key_prefix.format(file_name.split('.')[0]))

    return


def match_text(text_boxes: List, target_df: pd.DataFrame, target_columns: List):
    mapping = gt_class_mapping()
    label_fields = []
    for col in target_columns:
        value_ids = []
        value_text = []
        for index, row in target_df.iterrows():
            if not isinstance(row[col], str) and math.isnan(row[col]):
                continue
            targets = str(row[col]).split(' ')
            if col == "item_fee":
                targets = ['{:.2f}'.format(float(t)) for t in targets]
                targets = targets + ["$" + t for t in targets]
            cls = mapping[col]
            if col == "date_of_service":
                value_ids.extend([t['id'] for t in text_boxes if any([is_date_matched(t['text'], i) for i in targets])])
                value_text.extend([t['text'] for t in text_boxes if any([is_date_matched(t['text'], i) for i in targets])])
            else:
                #value_ids = [t['id'] for t in text_boxes if any([is_text_matched(t['text'], i) for i in targets])]
                #value_text = [t['text'] for t in text_boxes if any([is_text_matched(t['text'], i) for i in targets])]
                value_ids.extend([t['id'] for t in text_boxes if t['text'] in targets])
                value_text.extend([t['text'] for t in text_boxes if t['text'] in targets])
            if value_ids:
                field_label = get_field_label(cls, value_ids, value_text)
                label_fields.append(field_label._asdict())

    return label_fields

    
    
def merge_label_fields(anno_json):
    label_fields = []
    for cls in label_classes:
        value_ids = []
        value_text = []
        for f in anno_json['fields']:
            if f['field_name'] == cls:
                value_ids.extend(f['value_id'])
                value_text.extend(f['value_text'])
        if value_ids:
            field_label = get_field_label(cls, value_ids, value_text)
            label_fields.append(field_label._asdict())
    anno_json['fields'] = label_fields
    return anno_json
        

