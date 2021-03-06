from typing import List
from collections import namedtuple
from format_textract import TextractBlock, get_bbox_from_polygon


TextBox = namedtuple("TextBox", ["id", "bbox", "text"])

FieldLabel = namedtuple("FieldLabel", ["field_name", "value_id", "value_text", "key_id", "key_text"])

GlobalAttribute = namedtuple("GlobalAttribute", ["file_id"])

InputData = namedtuple("InputData", ["text_boxes", "fields", "global_attributes"])


label_classes = ["ServiceDate", "ProviderNum", "ItemNum", "ItemCharge", "ToothId"]


def format_text_boxes(blocks: TextractBlock, image_w:int, image_h:int) -> List[TextBox]:
    boxes = []
    for ind, bl in enumerate(blocks):
        # bbox - x_min, y_min, x_max, y_max
        bbox = get_bbox_from_polygon(bl.Polygon)
        bbox = [bbox[0] * image_w, bbox[1] * image_h, bbox[2] * image_w, bbox[3] * image_h]
        boxes.append(TextBox(ind, bbox, bl.Text))
    return boxes
        

def get_global_attribute(file_id):
    return GlobalAttribute(file_id)


def generate_input_data(text_boxes, fields, global_attributes):
    return InputData(text_boxes, fields, global_attributes)


def get_field_label(field_name, value_id, value_text, key_id=[], key_text=[]):
    return FieldLabel(field_name, value_id, value_text, key_id, key_text)

