default_scope = 'mmpose'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='posetrack18/Total AP',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False))
custom_hooks = [dict(type='SyncBuffersHook')]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-384x288-c161b7de_20220915.pth'
resume = False
file_client_args = dict(backend='disk')
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()
optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.0005))
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        milestones=[10, 15],
        gamma=0.1,
        by_epoch=True)
]
auto_scale_lr = dict(base_batch_size=512)
codec = dict(
    type='AssociativeEmbedding',
    input_size=(960, 480),
    heatmap_size=(960, 480),
    sigma=3)
num_things_classes = 1
num_stuff_classes = 0
num_classes = 1
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384)))),
    head=dict(
        type='PoseMask2FormerHead',
        keypoint_out_channels=17,
        in_channels=[256, 512, 1024, 2048],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=1,
        num_stuff_classes=0,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='mmdet.DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='mmdet.SinePositionalEncoding',
                num_feats=128,
                normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='mmdet.SinePositionalEncoding', num_feats=128,
            normalize=True),
        transformer_decoder=dict(
            type='mmdet.DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='mmdet.DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='mmdet.MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0, 0.1]),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=True))
dataset_type = 'PoseTrack18Dataset'
data_mode = 'bottomup'
data_root = 'data/posetrack18/'
train_pipeline = [
    dict(type='LoadImage', file_client_args=dict(backend='disk')),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='BottomupResize', input_size=(960, 480), resize_mode='expand'),
    dict(
        type='GenerateTarget',
        target_type='heatmap+keypoint_label',
        encoder=dict(
            type='AssociativeEmbedding',
            input_size=(960, 480),
            heatmap_size=(960, 480),
            sigma=3)),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', file_client_args=dict(backend='disk')),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(960, 480)),
    dict(type='PackPoseInputs')
]
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='PoseTrack18Dataset',
        data_root='data/posetrack18/',
        data_mode='bottomup',
        ann_file='annotations/posetrack18_train.json',
        data_prefix=dict(img=''),
        pipeline=[
            dict(type='LoadImage', file_client_args=dict(backend='disk')),
            dict(type='GetBBoxCenterScale'),
            dict(type='RandomFlip', direction='horizontal'),
            dict(
                type='BottomupResize',
                input_size=(960, 480),
                resize_mode='expand'),
            dict(
                type='GenerateTarget',
                target_type='heatmap+keypoint_label',
                encoder=dict(
                    type='AssociativeEmbedding',
                    input_size=(960, 480),
                    heatmap_size=(960, 480),
                    sigma=3)),
            dict(type='PackPoseInputs')
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='PoseTrack18Dataset',
        data_root='data/posetrack18/',
        data_mode='bottomup',
        ann_file='annotations/posetrack18_val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=[
            dict(type='LoadImage', file_client_args=dict(backend='disk')),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(960, 480)),
            dict(type='PackPoseInputs')
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='PoseTrack18Dataset',
        data_root='data/posetrack18/',
        data_mode='bottomup',
        ann_file='annotations/posetrack18_val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=[
            dict(type='LoadImage', file_client_args=dict(backend='disk')),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(960, 480)),
            dict(type='PackPoseInputs')
        ]))
val_evaluator = dict(
    type='PoseTrack18Metric',
    ann_file='data/posetrack18/annotations/posetrack18_val.json')
test_evaluator = dict(
    type='PoseTrack18Metric',
    ann_file='data/posetrack18/annotations/posetrack18_val.json')
launcher = 'none'
work_dir = './tmp/2'
