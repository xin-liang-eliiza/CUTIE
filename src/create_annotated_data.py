from utils import display_image_from_s3, read_json_from_s3, write_json_to_s3, write_image_to_s3
from format_textract import get_textract_blocks_by_type
from format_data import format_text_boxes, get_global_attribute, generate_input_data


#file_id = "0318922720232952_0.jpg"

im_bucket = "nib-prod-analytics-privacy-landing-ap-southeast-2"
im_key_prefix = "claim-receipt-recognition/category_id_19/{}"

transform_bucket = "nib-prod-analytics-privacy-transformed-ap-southeast-2"
textract_key_prefix = "claim-receipt-recognition/textract_raw_results/2020/01/category_id_19/{}_textract.json"
anno_text_key_prefix = "claim-receipt-recognition/annotations/text/category_id_19/{}_annotation.json"
anno_im_key_prefix = "claim-receipt-recognition/annotations/images/category_id_19/{}"



def annotate_data(file_ids):
    for fid in file_ids:
        try:
            print("Annotate data for file id {}".format(fid))
            # Generate annotated text data
            t_key = textract_key_prefix.format(fid)
            json_data = read_json_from_s3(transform_bucket, t_key)
            blocks = get_textract_blocks_by_type(json_data)

            i_key = im_key_prefix.format(fid)
            im = display_image_from_s3(im_bucket, i_key, displayed=False)
            text_boxes = format_text_boxes(blocks, im.width, im.height)
            global_attr = get_global_attribute(fid)

            input_data = generate_input_data([t._asdict() for t in text_boxes], [], global_attr._asdict())
            write_json_to_s3(input_data._asdict(), transform_bucket, anno_text_key_prefix.format(fid))

            # Generate annotated image data
            im = display_image_from_s3(im_bucket, i_key, text_boxes, displayed=False)
            write_image_to_s3(im, transform_bucket, anno_im_key_prefix.format(fid))
        except Exception as e:
            print("Error occurred for file id {}: {}".format(fid, repr(e)))
