from typing import List, Dict
import json
from collection import namedtuple


TextractBlock = namedtuple("TextractBlock", ["BlockType", "Confidence", "Text", "Geometry", "Polygon"]


def get_textract_blocks_by_type(json_data: Dict, bl_type: str = "WORD") -> List[|TextractBlock]:
    blocks = json_data.get("Blocks", [])
    outputs = []
    for bl in blocks:
        if bl.get("BlockType", "") != bl_type:
            continue
        outputs.append(TextractBlock(
            bl["BlockType"],
            bl["Confidence"],
            bl["Text"],
            bl["Geometry"],
            bl["Polygon"]))
    
    return outputs


def get_bbox_from_polygon(polygon: List) -> List:
    Xs = [i['X'] for i in polygon]
    Ys = [j['Y'] for j in polygon]
    bbox = [min(Xs), min(Ys), max(Xs), max(Ys)]
    return bbox
