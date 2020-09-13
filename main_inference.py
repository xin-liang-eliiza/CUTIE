import os, csv, timeit
import shutil
import tensorflow as tf
import numpy as np
import argparse
import json
from collections import namedtuple
from typing import Optional, List, Tuple, Dict
import pprint
import hdbscan
from sklearn.cluster import MeanShift

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from model_cutie_aspp import CUTIERes as CUTIEv1
from model_cutie2_aspp import CUTIE2 as CUTIEv2
from data_loader_json import DataLoader
from utils import *

from src.utils import *
from src.format_data import TextBox


item_fields = ["ServiceDate", "ItemNum", "ItemCharge", "ToothId"]
non_item_fields = ["ProviderNum"]

Prediction = namedtuple("Prediction", ["field_name", "text_box", "confidence"])
ExtractedField = namedtuple("ExtractedField", ["name", "value", "confidence"])


class Config:
    def __init__(self, doc_path):
        self.doc_path = doc_path

        self.use_cutie2 = False
        self.is_table = False
        self.test_path = ''
        self.fill_bbox = False
        self.positional_mapping_strategy = 1
        self.rows_target = 80
        self.cols_target = 80
        self.rows_ulimit = 80
        self.cols_ulimit = 80

        self.load_dict = True
        self.tokenize = True
        self.text_case = False
        self.dict_path = "dict/---" # not used if load_dict is True
        self.restore_ckpt = True

        self.embedding_size = 256
        self.batch_size = 1
        self.c_threshold = 0.5

        self.classes = ['DontCare', "ServiceDate", "ProviderNum", "ItemNum", "ItemCharge", "ToothId"]
        self.save_prefix = "DENTAL"
        self.e_ckpt_path = "graph/"
        self.ckpt_file = "CUTIE_atrousSPP_d20000c6(r80c80)_iter_900.ckpt"
        self.load_dict_from_path = "dict/DENTAL"


def inference_input(doc_path):
    cfg = Config(doc_path)
    args = cfg.__dict__
    print("Model configs", args)
    
    params = argparse.Namespace(**args)
    return params


def load_model(doc_path="inference_data"):
    params = inference_input(doc_path)

    data_loader = DataLoader(params, params.classes, update_dict=False, load_dictionary=True, data_split=0.0) # False to provide a path with only test data
    num_words = max(20000, data_loader.num_words)
    num_classes = data_loader.num_classes
    # model
    if params.use_cutie2:
        network = CUTIEv2(num_words, num_classes, params)
    else:
        network = CUTIEv1(num_words, num_classes, params)
    model_output = network.get_output('softmax')
    
    # evaluation
    ckpt_saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    try:
        ckpt_path = os.path.join(params.e_ckpt_path, params.save_prefix, params.ckpt_file)
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        print('Restoring from {}...'.format(ckpt_path))
        ckpt_saver.restore(sess, ckpt_path)
        print('{} restored'.format(ckpt_path))
    except:
        raise Exception('Check your pretrained {:s}'.format(ckpt_path))
    return network, model_output, sess


network, model_output, sess = load_model()


def infer(doc_path, network=network, model_output=model_output, sess=sess) -> List[Prediction]:
    params = inference_input(doc_path)

    data_loader = DataLoader(params, params.classes, update_dict=False, load_dictionary=True, data_split=0.0) # False to provide a path with only test data
    '''
    num_words = max(20000, data_loader.num_words)
    num_classes = data_loader.num_classes

    # model
    if params.use_cutie2:
        network = CUTIEv2(num_words, num_classes, params)
    else:
        network = CUTIEv1(num_words, num_classes, params)
    model_output = network.get_output('softmax')
    
    # evaluation
    ckpt_saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        try:
            ckpt_path = os.path.join(params.e_ckpt_path, params.save_prefix, params.ckpt_file)
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            print('Restoring from {}...'.format(ckpt_path))
            ckpt_saver.restore(sess, ckpt_path)
            print('{} restored'.format(ckpt_path))
        except:
            raise Exception('Check your pretrained {:s}'.format(ckpt_path))
   ''' 
    num_test = len(data_loader.validation_docs)
    results = []
    result_files = []
    for i in range(num_test):
        predictions = []
        data = data_loader.fetch_validation_data()
        print('{:d} samples left to be tested'.format(num_test-i))
        
#             grid_table = data['grid_table']
#             gt_classes = data['gt_classes']
        feed_dict = {
            network.data_grid: data['grid_table'],
        }
        if params.use_cutie2:
            feed_dict = {
                network.data_grid: data['grid_table'],
                network.data_image: data['data_image'],
                network.ps_1d_indices: data['ps_1d_indices']
            }
        fetches = [model_output]
        
        print(data['file_name'][0])
        print(data['grid_table'].shape, data['data_image'].shape, data['ps_1d_indices'].shape)
        
        timer_start = timeit.default_timer()
        [model_output_val] = sess.run(fetches=fetches, feed_dict=feed_dict)
        timer_stop = timeit.default_timer()
        print('\t >>time per step: %.2fs <<'%(timer_stop - timer_start))


        # visualize result
        shape = data['shape']
        file_name = data['file_name'][0] # use one single file_name
        bboxes = data['bboxes'][file_name]
        if not params.is_table:
            predictions = get_predicted_bboxes(data_loader, params.doc_path, np.array(data['grid_table'])[0], 
                     np.array(data['gt_classes'])[0], np.array(model_output_val)[0], file_name, np.array(bboxes), shape)
            results.append(predictions)
            result_files.append(file_name)
    return results, result_files


def get_predicted_bboxes(data_loader, file_prefix, grid_table, gt_classes, model_output_val, file_name,  bboxes, shape):
    data_input_flat = grid_table.reshape([-1])
    labels = gt_classes.reshape([-1])
    logits = model_output_val.reshape([-1, data_loader.num_classes])
    bboxes = bboxes.reshape([-1])
    
    #max_len = 768*2 # upper boundary of image display size 
    #img = cv2.imread(join(file_prefix, file_name))
    anno_json_path = os.path.join("eval_data", file_name.split('.')[0]+"_auto_label.json")
    text_boxes = json.load(open(anno_json_path))["text_boxes"]

    predictions = []
    existing_bboxes = []
    for i in range(len(data_input_flat)):
        if max(logits[i]) > c_threshold:
            if len(bboxes[i]) > 0:
                x_, y_, w_, h_ = bboxes[i]
                inf_id = np.argmax(logits[i])
                text_box = get_overlap_bbox([x_, y_, x_+w_, y_+h_], text_boxes)
                if text_box is None:
                    continue 
                text_box = TextBox(**text_box)
                # Avoid duplicate results
                if text_box.bbox not in existing_bboxes:
                    pd_class = data_loader.classes[inf_id]
                    predictions.append(Prediction(pd_class, text_box, float(max(logits[i]))))
                    existing_bboxes.append(text_box.bbox)

    return predictions


def post_processing(predictions):
    sanitised_predictions = [sanitise_prediction(p) for p in predictions if sanitise_prediction(p) is not None]
    date_predictions = [p for p in sanitised_predictions if p.field_name == "ServiceDate"]
    non_item_fields_new = non_item_fields
    if len(date_predictions) == 1:
        non_item_fields_new.append("ServiceDate")
    non_cluster_predictions = [p for p in sanitised_predictions if p.field_name in non_item_fields_new]
    predictions_to_cluster = [p for p in sanitised_predictions if p.field_name in item_fields]
    clusters = cluster_prediction(predictions_to_cluster)
    
    final_predictions = parse_final_results(non_cluster_predictions, clusters)
    return final_predictions


def parse_final_results(non_clusters: List[Prediction], clusters: Dict):
    final_results = [] 
    general_field_dict = {}
    for p in non_clusters:
        general_field_dict.update({p.field_name:
            ExtractedField(p.field_name, p.text_box.text, p.confidence)
        })
    for key, values in clusters.items():
        item_dict = {}
        sorted_values = sorted(values, key=lambda x: x.text_box.id)
        if key == -1 and len(clusters) > 1:
            continue
        for v in sorted_values:
            if v.field_name not in item_dict: 
                item_dict.update({v.field_name: ExtractedField(v.field_name, v.text_box.text, v.confidence)})
        item_dict.update(general_field_dict)
        final_results.append(item_dict)
    return final_results


def cluster_prediction(predictions):
    clustered_predictions = {}
    
    # Coarse clustering
    data_to_cluster = get_clustering_data_from_predictions(predictions)
    clusters = run_clustering(data_to_cluster, method="meanshift")
    
    major_cluster_thresh = 3
    unique_clusters, unique_counts = np.unique(clusters, return_counts=True)
    major_cluster_ids = unique_clusters[np.where(unique_counts > major_cluster_thresh)]

    clusters_v1 = [predictions[x] for i in major_cluster_ids for x in np.where(clusters == i)[0]]
    predictions = clusters_v1

    # Fine-grain clustering
    data_to_cluster = get_clustering_data_from_predictions(predictions)
    clusters = run_clustering(data_to_cluster, method="hdbscan")
    unique_clusters = np.unique(clusters)
    for i in unique_clusters:
        clustered_predictions[i] = [predictions[x] for x in np.where(clusters == i)[0]]
        print("CLUCSTER ", i)
        print(json.dumps(clustered_predictions[i], indent=4))
    
    return clustered_predictions


def get_clustering_data_from_predictions(predictions):
    data = []
    for p in predictions:
        centroid = get_bbox_centroid(p.text_box.bbox)
        data.append((p.text_box.id, centroid[1]))
    #data = np.array(data).reshape(-1, 1)
    data = np.array(data)
    print("Cluster data: ", data)

    return data


def run_clustering(data, method="meanshift"):
    clusters = []
    if method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(algorithm='best', min_cluster_size=2)
        clusterer.fit(data)
        clusters = clusterer.labels_ 
    elif method == "meanshift":
        ms = MeanShift(bandwidth=None, bin_seeding=False)
        ms.fit(data)
        clusters = ms.labels_
    print("Clusters: ", clusters)

    return clusters


def sanitise_prediction(prediction: Prediction) -> Optional[Prediction]:
    result = None
    if prediction.field_name == "ServiceDate":
        if is_valid_date(prediction.text_box.text):
            result = prediction
    elif prediction.field_name == "ProviderNum":
        provider_num = validate_provider(prediction.text_box.text)
        if provider_num is not None:
            text_box = prediction.text_box._replace(text=provider_num)
            result = prediction._replace(text_box=text_box)
    elif prediction.field_name == "ItemNum":
        if is_valid_item_num(prediction.text_box.text):
            result = prediction
    elif prediction.field_name == "ItemCharge":
        item_charge = validate_charge(prediction.text_box.text)
        if item_charge is not None:
            text_box = prediction.text_box._replace(text=item_charge)
            result = prediction._replace(text_box=text_box)
    return result
    

if __name__ == "__main__":
    #doc_path = "inference_data/"
    doc_path = "inference_single_data/"
    inference_dict = {}
    results, result_files = infer(doc_path)
    for ind, r in enumerate(results):       
        print("{} th result".format(ind))
        try:
            print("RESULTS for ", result_files[ind])
            results = post_processing(r)
            print("FINAL RESULTS for", result_files[ind])
            print(json.dumps(results, indent=4))
            inference_dict[result_files[ind]] = results
        except Exception as e:
            print(e)
    #json.dump(inference_dict, open("inference_results.json", "w"), indent=4)

    
