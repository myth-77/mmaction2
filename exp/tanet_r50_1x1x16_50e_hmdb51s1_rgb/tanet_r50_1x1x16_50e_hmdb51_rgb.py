model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='TANet',
        pretrained='torchvision://resnet50',
        depth=50,
        num_segments=16,
        tam_cfg=dict()),
    cls_head=dict(
        type='TSMHead',
        num_classes=51,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.6,
        init_std=0.001,
        num_segments=16),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
checkpoint_config = dict(interval=5)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
optimizer = dict(
    type='SGD',
    constructor='TSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(policy='step', step=[30, 40, 45])
total_epochs = 50
split = 1
dataset_type = 'RawframeDataset'
data_root = 'data/hmdb51/rawframes'
data_root_val = 'data/hmdb51/rawframes'
ann_file_train = 'data/hmdb51/hmdb51_train_split_1_rawframes.txt'
ann_file_val = 'data/hmdb51/hmdb51_val_split_1_rawframes.txt'
ann_file_test = 'data/hmdb51/hmdb51_val_split_1_rawframes.txt'
sthv1_flip_label_map = dict({2: 4, 4: 2, 30: 41, 41: 30, 52: 66, 66: 52})
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(
        type='Flip',
        flip_ratio=0.5,
        flip_label_map=dict({
            2: 4,
            4: 2,
            30: 41,
            41: 30,
            52: 66,
            66: 52
        })),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        twice_sample=True,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RawframeDataset',
        ann_file='data/hmdb51/hmdb51_train_split_1_rawframes.txt',
        data_prefix='data/hmdb51/rawframes',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=16),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(
                type='MultiScaleCrop',
                input_size=224,
                scales=(1, 0.875, 0.75, 0.66),
                random_crop=False,
                max_wh_scale_gap=1,
                num_fixed_crops=13),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(
                type='Flip',
                flip_ratio=0.5,
                flip_label_map=dict({
                    2: 4,
                    4: 2,
                    30: 41,
                    41: 30,
                    52: 66,
                    66: 52
                })),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    val=dict(
        type='RawframeDataset',
        ann_file='data/hmdb51/hmdb51_val_split_1_rawframes.txt',
        data_prefix='data/hmdb51/rawframes',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=16,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]),
    test=dict(
        type='RawframeDataset',
        ann_file='data/hmdb51/hmdb51_val_split_1_rawframes.txt',
        data_prefix='data/hmdb51/rawframes',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=16,
                twice_sample=True,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='ThreeCrop', crop_size=256),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
work_dir = './exp/tanet_r50_1x1x16_50e_hmdb51s1_rgb'
gpu_ids = range(0, 2)
omnisource = False
module_hooks = []
