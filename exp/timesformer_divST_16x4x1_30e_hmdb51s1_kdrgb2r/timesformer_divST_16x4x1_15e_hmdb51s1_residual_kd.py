checkpoint_config = dict(interval=5)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = ''
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
model_teacher = dict(
    backbone=dict(
        type='TimeSformer',
        num_frames=16,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.0,
        transformer_layers=None,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-06)),
    cls_head=dict(type='TimeSformerHead', num_classes=51, in_channels=768),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
model = dict(
    type='Recognizer3Dkd',
    backbone=dict(
        type='TimeSformer',
        pretrained=
        'https://download.openmmlab.com/mmaction/recognition/timesformer/vit_base_patch16_224.pth',
        num_frames=16,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.0,
        transformer_layers=None,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-06)),
    teacher=dict(
        backbone=dict(
            type='TimeSformer',
            num_frames=16,
            img_size=224,
            patch_size=16,
            embed_dims=768,
            in_channels=3,
            dropout_ratio=0.0,
            transformer_layers=None,
            attention_type='divided_space_time',
            norm_cfg=dict(type='LN', eps=1e-06)),
        cls_head=dict(type='TimeSformerHead', num_classes=51, in_channels=768),
        train_cfg=None,
        test_cfg=dict(average_clips='prob')),
    teacher_path=
    '/home/myth/workplace/mmaction2/exp/timesformer_divST_16x4x1_15e_hmdb51s1_rgb_SGD1e4_finetunek400/best_top1_acc_epoch_15.pth',
    cls_head=dict(type='TimeSformerHead', num_classes=51, in_channels=768),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'),
    loss_kd=dict(type='CrossEntropyLoss', loss_weight=1.0))
split = 1
dataset_type = 'RawframeDataset'
data_root = 'data/hmdb51/rawframes'
data_root_val = 'data/hmdb51/rawframes'
ann_file_train = 'data/hmdb51/hmdb51_train_split_1_rawframes.txt'
ann_file_val = 'data/hmdb51/hmdb51_val_split_1_rawframes.txt'
ann_file_test = 'data/hmdb51/hmdb51_val_split_1_rawframes.txt'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
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
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
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
                clip_len=16,
                frame_interval=4,
                num_clips=1),
            dict(type='RawFrameDecode'),
            dict(type='RandomRescale', scale_range=(256, 320)),
            dict(type='RandomCrop', size=224),
            dict(type='Flip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        start_index=1,
        modality='Residual'),
    val=dict(
        type='RawframeDataset',
        ann_file='data/hmdb51/hmdb51_val_split_1_rawframes.txt',
        data_prefix='data/hmdb51/rawframes',
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
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        start_index=1,
        modality='Residual'),
    test=dict(
        type='RawframeDataset',
        ann_file='data/hmdb51/hmdb51_val_split_1_rawframes.txt',
        data_prefix='data/hmdb51/rawframes',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=16,
                frame_interval=4,
                num_clips=1,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 224)),
            dict(type='ThreeCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        start_index=1,
        modality='Residual'))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
optimizer = dict(
    type='SGD',
    lr=0.000125,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys=dict({
            '.backbone.cls_token': dict(decay_mult=10.0),
            '.backbone.pos_embed': dict(decay_mult=10.0),
            '.backbone.time_embed': dict(decay_mult=10.0)
        })),
    weight_decay=0.0001,
    nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[15, 20])
total_epochs = 30
work_dir = './exp/timesformer_divST_16x4x1_30e_hmdb51s1_kdrgb2r'
gpu_ids = range(0, 1)
omnisource = False
module_hooks = []
