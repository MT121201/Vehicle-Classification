_base_ = "../mmpretrain/repvgg/repvgg-A0_8xb32_in1k.py"

# classes
class_name = ('call', 'read', 'purchase', 'chatting')
num_classes = len(class_name)
metainfo = dict(classes=class_name)

# Load pretrained model from github
load_from = "https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_8xb32_in1k_20221213-60ae8e23.pth"

data_preprocessor = dict(
    num_classes=num_classes,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

# dataset
dataset_type = 'ImageNet'
train_pipeline = [
    dict(type='LoadSuperImage', super_image_grid=(3,3), track_info_txt='/data/its/oad/triet_test/annotations/track_frame.txt',frame_per_group=16),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
    dict(type='GaussianBlur', prob=0.5, radius=2 ),
    dict(type ='Lighting', eigval=[0.2175, 0.0188, 0.0045], eigvec=[[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]],),
    dict(type='Rotate', angle= 20 ,prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomGrayscale', prob=0.2, keep_channels=True),
    # # dict(
    # #     type='RandomErasing',
    # #     erase_prob=0.2,
    # #     mode='rand',
    # #     min_area_ratio=0.02,
    # #     max_area_ratio=1 / 4,
    # #     fill_color=bgr_mean,
    # #     fill_std=bgr_std),
    # dict(type='RandomResize', scale=(672,672), ratio_range=(0.8,1.2)),
    # dict(type='RandomCrop', crop_size=224, padding=(3,3,3,3), pad_if_needed=True),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadSuperImage', super_image_grid=(3,3), track_info_txt='/data/its/oad/triet_test/annotations/track_frame.txt',frame_per_group=16),
    # dict(type='ResizeEdge', scale=672, edge='short', backend='pillow'),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

dataset_1_train = dict(
    type=dataset_type,
    data_root='/data/its/oad/triet_test/',
    metainfo=metainfo,
    ann_file='annotations/train.txt',
    data_prefix='cropped_image',
    with_label=True,
    pipeline=train_pipeline
)

# repeat dataset
dataset_1_train_repeat = dict(
    type='RepeatDataset',
    times=40,
    dataset=dataset_1_train
)


train_dataloader = dict(
    batch_size=256,
    num_workers=5,
    dataset=dataset_1_train_repeat,
    _delete_=True,
)

# Val dataloaders
dataset_1_val = dict(
    type=dataset_type,
    data_root='/data/its/oad/triet_test/',
    metainfo=metainfo,
    ann_file='annotations/test.txt',
    data_prefix='cropped_image',
    with_label=True,
    pipeline=test_pipeline
)

val_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dataset_1_val
)

# accuracy
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# OR Precision, Recall, F1-score
# val_evaluator = dict(
#     _delete_=True,
#     type='SingleLabelMetric',
#     average=None, # Print out classwise
#     # average='macro' # average
# )

test_dataloader = val_dataloader
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=2)
default_hooks = dict(checkpoint=dict(type='CheckpointHook',
                     interval=10, max_keep_ckpts=2, save_best='auto'))

# fp16 training help you x2 batch size
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale=512.0)