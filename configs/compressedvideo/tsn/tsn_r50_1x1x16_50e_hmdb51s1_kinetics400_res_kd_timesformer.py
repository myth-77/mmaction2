model_teacher = dict(
    backbone=dict(
        type='TimeSformer',
        num_frames=16,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.,
        transformer_layers=None,
    attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(type='TimeSformerHead', num_classes=51, in_channels=768),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
model = dict(
    type='Recognizer3Dkd_RBG2Res',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=51,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01),
    teacher = model_teacher,
    teacher_path = '/home/myth/workplace/mmaction2/exp/timesformer_divST_16x4x1_15e_hmdb51s1_rgb_SGD1e4_finetunek400/best_top1_acc_epoch_15.pth',
    train_cfg=None,
    test_cfg=dict(average_clips='prob'),
    weight_loss = (0.9, 0.1))
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 50
checkpoint_config = dict(interval=5)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x8_100e_kinetics400_rgb/tsn_r50_256p_1x1x8_100e_kinetics400_rgb_20200817-883baf16.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
split = 1
dataset_type = 'RawframeDataset'
data_root = 'data/hmdb51/rawframes'
data_root_val = 'data/hmdb51/rawframes'
ann_file_train = 'data/hmdb51/hmdb51_train_split_1_rawframes.txt'
ann_file_val = 'data/hmdb51/hmdb51_val_split_1_rawframes.txt'
ann_file_test = 'data/hmdb51/hmdb51_val_split_1_rawframes.txt'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
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
    dict(type='CenterCrop', crop_size=256),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
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
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RawframeDataset',
        ann_file='data/hmdb51/hmdb51_train_split_1_rawframes.txt',
        data_prefix='data/hmdb51/rawframes',
        pipeline=[
            dict(
                type='SampleFrames', clip_len=1, frame_interval=1,
                num_clips=16),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='RandomResizedCrop'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        modality='RGB_RES'),
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
            dict(type='CenterCrop', crop_size=256),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ],
        modality='RGB_RES'),
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
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=256),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ],
        modality='RGB_RES'))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
work_dir = './exp/tsn_r50_1x1x8_50e_hmdb51s1_k400pre_res'
gpu_ids = range(0, 1)
omnisource = False
module_hooks = []
