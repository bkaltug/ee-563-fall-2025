# YOLO v3 configuration files.
_base_ = [
    'mmdetection/configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py',
]

data_root = '../dataset/'  

metainfo = {
    'classes': ('balloon',),
    'palette': [
        (220, 20, 60),
    ]
}

train_dataloader = dict(
    batch_size=2, 
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/') 
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',  
        data_prefix=dict(img='val/') 
    )
)

test_dataloader = val_dataloader 

model = dict(
    bbox_head=dict(
        num_classes=1 
    )
)

load_from = load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'

train_cfg = dict(max_epochs=50, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, milestones=[35, 45], gamma=0.1)
]

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric='bbox')

test_evaluator = val_evaluator