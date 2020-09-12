import math
import pandas
import json
from collections import Counter, namedtuple
from dateutil.parser import parse as dt_parse

from utils import read_csv_from_s3, format_date


gt_columns = ['photo_tracking_number', 'category_id', 'member_number', 'page_number', 'provider_number.1', 'claim_item_number', 'claim_item_entry', 'tooth_code', 'date_of_service', 'item_fee', 'script_number']
pd_columns = ["photo_tracking_number", "page_number", "ProviderNum", "ItemNum", "ServiceDate", "ItemCharge", "ToothId"]

Accuracy = namedtuple("Accuracy", ["field", "f1"])


def get_true_positivies(l1, l2):
    tps = list((Counter(l1) & Counter(l2)).elements()) 
    return tps


def calc_precision(tp, fp):
    return (tp / (tp + fp))


def calc_recall(tp, fn):
    return (tp / (tp + fn))


def calc_f1_score(precision, recall):
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calc_field_accuracy(gt_df, pd_df):
    gts = gt_df.tolist()
    pds = pd_df.tolist()
    tp = len(get_true_positivies(gts, pds))
    fp = len(pds) - tp
    fn = len(gts) - tp
    print("tp, fp, fn")
    print(tp, fp, fn)
    
    prec = calc_precision(tp, fp)
    rec = calc_recall(tp, fn)
    print("prec, rec")
    print(prec, rec)
    if prec == 0 and rec == 0:
        f1 = 0
    else:
        f1 = calc_f1_score(prec, rec)
    return f1


def calc_accuracies(gt_df, pd_df, fields=[]):
    results = []
    for field in fields:
        print("{} accuracy".format(field))
        f1 = calc_field_accuracy(gt_df[field], pd_df[field])
        results.append(Accuracy(field, f1))
    return results


def prediction_class_mapping():
    mapping = {
        "ProviderNum": "provider_number.1",
        "ItemNum": "claim_item_number",
        "ServiceDate": "date_of_service",
        "ItemCharge": "item_fee",
        "ToothId": "tooth_code"
    }
    return mapping


def pd_json_to_df(pd_json, pd_columns=pd_columns):
    pd = json.load(open(pd_json))
    pd_cls_mapping = prediction_class_mapping()
    pd_df = pandas.DataFrame(columns=list(pd_cls_mapping.values()))
    for key, values in pd.items():
        file_id = key.split('.')[0]
        photo_tracking_num, page_num = file_id.split("_")
        for val in values:
            item_dict = {j: val.get(i, [None, None, None])[1] for i, j in pd_cls_mapping.items()}
            item_dict.update({"photo_tracking_number": photo_tracking_num, "page_number": float(page_num)})
            pd_df = pd_df.append(item_dict, ignore_index=True)
    pd_df['date_of_service'] = pd_df['date_of_service'].apply(lambda x: format_date(x) if format_date(x) else float('nan'))
    pd_df['provider_number.1'] = pd_df['provider_number.1'].apply(lambda x: x.strip("(").strip(")") if isinstance(x, str) else x)
    return pd_df

    
    
def run(gt_bucket, gt_csv_key, pd_json, gt_columns=gt_columns):
    gt_df = read_csv_from_s3(gt_bucket, gt_csv_key) 
    gt_df = gt_df[gt_columns]
    pd_df = pd_json_to_df(pd_json)
    pd_df = pd_df.drop_duplicates()
    print(pd_df)
    pd_df.to_csv("predictions.csv")

    gt_df = gt_df[gt_df["photo_tracking_number"].isin(pd_df["photo_tracking_number"])]
    gt_df = gt_df.dropna(subset=['date_of_service'])
    gt_df = gt_df.drop_duplicates()
    print(gt_df)
    gt_df_new = pandas.DataFrame()
    for ptn in list(set(pd_df["photo_tracking_number"].tolist())):
        gt_temp = gt_df[gt_df["photo_tracking_number"] == float(ptn)]
        pd_temp = pd_df[pd_df["photo_tracking_number"] == ptn]
        gt_temp = gt_temp[gt_temp["page_number"].isin(pd_temp["page_number"])]
        gt_df_new = gt_df_new.append(gt_temp, ignore_index=True)
    print(gt_df_new)
    gt_df_new.to_csv("ground_truth.csv")
    pd_cls_mapping = prediction_class_mapping()
    results = calc_accuracies(gt_df_new, pd_df, list(pd_cls_mapping.values())) 
    
    return results


if __name__ == "__main__":
    bucket = "nib-prod-analytics-privacy-landing-ap-southeast-2"
    key = "claim-receipt-recognition/CAS submissions 2019-11-01 to 2020-01-12.csv"
    pd_json = "/home/ec2-user/SageMaker/CUTIE/inference_results.json"
    results = run(bucket, key, pd_json)
    print(results)


