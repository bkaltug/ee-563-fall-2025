import os
import os.path
import numpy as np
import mmcv
import shutil
from mmdet.apis import DetInferencer
from pycocotools import mask as mask_util

models_to_test = [
    {
        'name': 'Mask_R-CNN',
        'config': 'balloon_mask_rcnn_config.py',
        'weights': 'mmdetection/work_dirs/balloon_mask_rcnn_config/epoch_3.pth'
    },
    {
        'name': 'YOLOv3',
        'config': 'balloon_yolo_config.py',
        'weights': 'mmdetection/work_dirs/balloon_yolo_config/epoch_42.pth'
    }
]

images_to_test = ['images/balloon_1.jpg', 'images/balloon_2.jpg', 'images/balloon_3.jpg']

output_dir = 'results'
vis_dir = os.path.join(output_dir, 'vis')
os.makedirs(output_dir, exist_ok=True)

print("Starting Inference & Area Calculation")

for model_info in models_to_test:
    model_name = model_info['name']
    config_path = model_info['config']
    weights_path = model_info['weights']
    
    print(f"\nInitializing {model_name}")
    
    inferencer = DetInferencer(
        model=config_path, 
        weights=weights_path, 
        device='cpu' 
    )

    for image_path in images_to_test:
        
        img = mmcv.imread(image_path)
        if img is None:
            print(f"ERROR: Could not read {image_path}. Skipping.")
            continue
            
        height, width, _ = img.shape
        total_pixels = height * width

        result_dict = inferencer(
            inputs=image_path, 
            out_dir=output_dir,
            no_save_vis=False 
        )
        
        predictions = result_dict['predictions'][0]
        total_balloon_area_pixels = 0
        
        if model_name == 'Mask_R-CNN':
            if 'masks' in predictions:
                for rle_mask in predictions['masks']: 
                    boolean_mask = mask_util.decode(rle_mask)
                    total_balloon_area_pixels += np.sum(boolean_mask)
        
        elif model_name == 'YOLOv3':
            if 'bboxes' in predictions:
                for bbox in predictions['bboxes']:
                    x1, y1, x2, y2 = bbox
                    bbox_area = (x2 - x1) * (y2 - y1)
                    total_balloon_area_pixels += (bbox_area / 2.0)
        
        area_percentage = (total_balloon_area_pixels / total_pixels) * 100
        

        image_filename = os.path.basename(image_path)
        
        print(f"Result for {image_filename}:")
        print(f"Total Balloon Area: {area_percentage:.2f}%")
        

        original_vis_path = os.path.join(vis_dir, image_filename)
        

        new_vis_path = os.path.join(output_dir, f"{model_name}_{image_filename}")
        
        if os.path.exists(original_vis_path):
            os.rename(original_vis_path, new_vis_path)
        else:
            print(f"Warning: Could not find output file at {original_vis_path} to rename.")
            
if os.path.exists(vis_dir):
    shutil.rmtree(vis_dir)
    print(f"\nRemoved empty {vis_dir} folder.")

print(f"Processed images can be found at '{output_dir}/' folder.")