model_teacher = dict(
    backbone=dict(
        type='TimeSformer',
        num_frames=16,
        in_channels=3,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        dropout_ratio=0.0,
        transformer_layers=None,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-06)),
    cls_head=dict(type='TimeSformerHead', num_classes=51, in_channels=768),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
model = dict(
    type='Recognizer3Dkd_RBG2Res_Feat',
    backbone=dict(
        type='ResNet3d',
        pretrained2d=True,
        pretrained='torchvision://resnet50',
        depth=50,
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
    teacher=dict(
        backbone=dict(
            type='TimeSformer',
            num_frames=16,
            in_channels=3,
            img_size=224,
            patch_size=16,
            embed_dims=768,
            dropout_ratio=0.0,
            transformer_layers=None,
            attention_type='divided_space_time',
            norm_cfg=dict(type='LN', eps=1e-06)),
        cls_head=dict(type='TimeSformerHead', num_classes=51, in_channels=768),
        train_cfg=None,
        test_cfg=dict(average_clips='prob')),
    teacher_path=
    '/home/myth/workplace/mmaction2/exp/timesformer_divST_16x4x1_15e_hmdb51s1_rgb_SGD1e4_finetunek400/best_top1_acc_epoch_15.pth',
    train_cfg=None,
    test_cfg=dict(average_clips='prob'),
    weight_loss=(0.9, 0.1))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[40, 80])
total_epochs = 100
checkpoint_config = dict(interval=5)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = ''
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
    mean=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
    to_bgr=False)
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
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
        to_bgr=False),
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
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
        to_bgr=False),
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
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=64,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RawframeDataset',
        ann_file='data/hmdb51/hmdb51_train_split_1_rawframes.txt',
        data_prefix='data/hmdb51/rawframes',
        modality='RGB_RES',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=16,
                frame_interval=4,
                num_clips=1),
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
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    val=dict(
        type='RawframeDataset',
        ann_file='data/hmdb51/hmdb51_val_split_1_rawframes.txt',
        data_prefix='data/hmdb51/rawframes',
        modality='RGB_RES',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=16,
                frame_interval=4,
                num_clips=1,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]),
    test=dict(
        type='RawframeDataset',
        ann_file='data/hmdb51/hmdb51_val_split_1_rawframes.txt',
        data_prefix='data/hmdb51/rawframes',
        modality='RGB_RES',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=16,
                frame_interval=4,
                num_clips=10,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='ThreeCrop', crop_size=256),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5, 127.5, 127.5, 127.5],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
work_dir = './exp/i3d_r50_16x4x1_hmdb51s1_timsformer_kd_feat_residual_cls9_kd1'
gpu_ids = range(0, 1)
omnisource = False
module_hooks = []