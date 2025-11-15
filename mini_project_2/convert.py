import os
import os.path as osp
import json
import mmengine
import mmcv

def convert_balloon_to_coco(ann_file, out_file, image_prefix):

    os.makedirs(osp.dirname(out_file), exist_ok=True)

    data_infos = json.load(open(ann_file))
    
    annotations = []
    images = []
    obj_count = 0
    

    for idx, v in enumerate(data_infos.values()):
        image_filename = v['filename']
        filepath = osp.join(image_prefix, image_filename)
        

        if not osp.exists(filepath):
            print(f"Warning: Image file not found {filepath}, skipping annotation.")
            continue
            
        height, width = mmcv.imread(filepath).shape[:2]

        images.append(dict(
            id=idx,
            file_name=image_filename,
            height=height,
            width=width))

        for i, region in enumerate(v['regions'].values()):
            shape_attributes = region['shape_attributes']
            
            if 'all_points_x' not in shape_attributes or 'all_points_y' not in shape_attributes:
                print(f"Warning: 'all_points_x' or 'all_points_y' not found in region for {image_filename}, skipping region.")
                continue
                
            px = shape_attributes['all_points_x']
            py = shape_attributes['all_points_y']
            

            if not px or not py:
                print(f"Warning: Empty polygon for {image_filename}, skipping region.")
                continue
                
            poly = [(x, y) for x, y in zip(px, py)]
            poly_flat = [p for pair in poly for p in pair]


            x_min = min(px)
            y_min = min(py)
            x_max = max(px)
            y_max = max(py)

            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            bbox_area = bbox_width * bbox_height 

            bbox = [x_min, y_min, bbox_width, bbox_height]

            data_ann = dict(
                image_id=idx,
                id=obj_count,
                category_id=0, 
                bbox=bbox,
                area=bbox_area,
                segmentation=[poly_flat],
                iscrowd=0)
            
            annotations.append(data_ann)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id': 0, 'name': 'balloon'}])
    
    mmengine.dump(coco_format_json, out_file)
    print(f"Successfully created COCO annotation: {out_file}")

# Traning annotations
convert_balloon_to_coco(
    ann_file='dataset/train/via_region_data.json',
    out_file='dataset/annotations/train.json',
    image_prefix='dataset/train')

# Validation annotations
convert_balloon_to_coco(
    ann_file='dataset/val/via_region_data.json',
    out_file='dataset/annotations/val.json',
    image_prefix='dataset/val')
