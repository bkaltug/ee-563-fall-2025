# Models Directory

Place your MMDetection model files here:

1. `faster_rcnn_r50_fpn_1x_coco.py` - Model configuration
2. `faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth` - Model weights

## Download Instructions

### Option 1: Using MIM (Recommended)

```bash
pip install -U openmim
mim download mmdet --config faster_rcnn_r50_fpn_1x_coco --dest ./
```

### Option 2: Manual Download

- **Config**: Download from [MMDetection GitHub](https://github.com/open-mmlab/mmdetection/blob/main/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py)
- **Weights**: Download from [OpenMMLab](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth)

## Custom Food Detection Models

For better food detection, consider training on:
- [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- [UECFOOD-256](http://foodcam.mobi/dataset256.html)
- [ISIA Food-500](https://github.com/ISIA-CNN/ISIA-Food-500)

Place your custom model files here and update the paths in `services/image_recognition.py`.
