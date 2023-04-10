_base_ = [
    '../../_base_/schedules/sgd_100e.py',
    '../../_base_/default_runtime.py'
]

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3d',
        pretrained2d=True,
        pretrained='torchvision://resnet50',
        depth=50,
        in_channels=5,
        conv1_kernel=(5, 7, 7),
        conv1_stride_t=2,
        pool1_stride_t=2,
        conv_cfg=dict(type='Conv3d'),
        norm_eval=False,
        inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        zero_init_residual=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=51,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# dataset settings
split = 1
dataset_type = 'RawframeDataset'
data_root = 'data/hmdb51/rawframes'
data_root_val = 'data/hmdb51/rawframes'
ann_file_train = f'data/hmdb51/hmdb51_train_split_{split}_rawframes.txt'
ann_file_val = f'data/hmdb51/hmdb51_val_split_{split}_rawframes.txt'
ann_file_test = f'data/hmdb51/hmdb51_val_split_{split}_rawframes.txt'

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5, 0, 0], std=[127.5, 127.5, 127.5, 1.0, 1.0], to_bgr=False)
    
train_pipeline = [
    dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=12,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        modality='RES_MV',
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        modality='RES_MV',
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        modality='RES_MV',
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/i3d_r50_32x2x1_100e_kinetics400_rgb/'
