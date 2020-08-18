import os, csv, timeit
import tensorflow as tf
import numpy as np
import argparse
import json
from collections import namedtuple
from typing import Optional, List
import pprint

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from model_cutie_aspp import CUTIERes as CUTIEv1
from model_cutie2_aspp import CUTIE2 as CUTIEv2
from data_loader_json import DataLoader
from utils import *

from src.utils import *
from src.format_data import TextBox


Prediction = namedtuple("Prediction", ["field_name", "text_box", "confidence"])


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


def infer(doc_path) -> List[Prediction]:
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
        
        num_test = len(data_loader.validation_docs)
        results = []
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
        return results


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
    for i in range(len(data_input_flat)):
        if max(logits[i]) > c_threshold:
            if len(bboxes[i]) > 0:
                x_, y_, w_, h_ = bboxes[i]
                inf_id = np.argmax(logits[i])
                text_box = get_overlap_bbox([x_, y_, x_+w_, y_+h_], text_boxes)
                text_box = TextBox(**text_box)
                pd_class = data_loader.classes[inf_id]
                predictions.append(Prediction(pd_class, text_box, max(logits[i])))

    return predictions


def post_processing(predictions):
    sanitised_predictions = [sanitise_prediction(p) for p in predictions if sanitise_prediction(p) is not None]
    final_predictions = sanitised_predictions
    #TODO geometry post processing
    return final_predictions


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
    doc_path = "inference_data/"
    results = infer(doc_path)
    results = post_processing(results[0])
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(results)
    print(json.dumps(results, indent=4))
    
