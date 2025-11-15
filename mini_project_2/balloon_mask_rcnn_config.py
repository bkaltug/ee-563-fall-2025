# Mask RCNN configuration files.
_base_ = [
    'mmdetection/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py',
]

data_root = '../dataset/'  
metainfo = {
    'classes': ('balloon',),  
    'palette': [
        (220, 20, 60),
    ]
}

train_dataloader = dict(
    batch_size=1, 
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
    roi_head=dict(
        bbox_head=dict(
            num_classes=1 
        ),
        mask_head=dict(
            num_classes=1
        )
    )
)

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric=['bbox', 'segm'])

test_evaluator = val_evaluator