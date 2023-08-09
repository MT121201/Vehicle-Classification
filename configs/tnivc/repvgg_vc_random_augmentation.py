_base_ = "../mmpretrain/repvgg/repvgg-A0_8xb32_in1k.py"

# classes
class_name = ('xe sedan', 'xe SUV', 'xe ban tai', 'xe ba gac', 'xe tai nho',
              'xe tai lon', 'xe container', 'xe may xuc, may cau, xe lu, xe cho be tong',
              'xe 16 cho', 'xe 29-32 cho', 'xe 52 cho')
num_classes = len(class_name)
metainfo = dict(classes=class_name)

# Load pretrained model from github
load_from = "https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_8xb32_in1k_20221213-60ae8e23.pth"

# Change head numclasses and loss weight
model = dict(
    head=dict(num_classes=num_classes,
              loss=dict(type='CrossEntropyLoss', loss_weight=1.0,
                        class_weight=[0.6362, 0.6810, 1, 1, 0.9137, 0.3440, 1, 1, 1, 1, 1])))
#Class          sedan   suv     bantai  bagac   tainho  tailon  container   cau+xuc=..  16cho   29cho   52cho
#After repeat	13070	12210	4100	1760	9100	24170	7250	    6650	    4100	2110	6940
#Balance factor 0.6362  0.6810  1       1       0.9137  0.3440  1           1           1       1       1

# True calculate class weight:
# Class 1 weight: 0.6362
# Class 2 weight: 0.6810
# Class 3 weight: 2.0279
# Class 4 weight: 4.7242
# Class 5 weight: 0.9137
# Class 6 weight: 0.3440
# Class 7 weight: 1.1468
# Class 8 weight: 1.2503
# Class 9 weight: 2.0279
# Class 10 weight: 3.9405
# Class 11 weight: 1.1981
# Change num classes in preprocessor
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
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

dataset_1_train = dict(
    type=dataset_type,
    data_root='/data/its/vehicle_cls/vp3_202307_crop/',
    metainfo=metainfo,
    ann_file='annotations/train.txt',
    data_prefix='',
    with_label=True,
    pipeline=train_pipeline
)

dataset_2_train = dict(
    type=dataset_type,
    data_root='/data/its/vehicle_cls/202307_crop_ttp/',
    metainfo=metainfo,
    ann_file='annotations/train.txt',
    # split='train',
    data_prefix='images/',
    with_label=True,
    pipeline=train_pipeline
)
dataset_3_train = dict(
    type=dataset_type,
    data_root='/data/its/vehicle_cls/vehicle_v3/',
    metainfo=metainfo,
    ann_file='annotations/train.txt',
    # split='train',
    data_prefix='images/',
    with_label=True,
    pipeline=train_pipeline)

dataset_4_train = dict(
    type=dataset_type,
    data_root='/data/its/vehicle_cls/vehicle_v4/',
    metainfo=metainfo,
    ann_file='annotations/train.txt',
    # split='train',
    data_prefix='images/',
    with_label=True,
    pipeline=train_pipeline)

# repeat dataset
dataset_1_train_repeat = dict(
    type='RepeatDataset',
    times=40,
    dataset=dataset_1_train
)
dataset_2_train_repeat = dict(
    type='RepeatDataset',
    times=10,
    dataset=dataset_2_train
)
dataset_3_train_repeat = dict(
    type='RepeatDataset',
    times=40,
    dataset=dataset_3_train)

dataset_4_train_repeat = dict(
    type='RepeatDataset',
    times=80,
    dataset=dataset_4_train)
# Concat dataset
dataset_concat = dict(
    type='ConcatDataset',
    _delete_=True,
    datasets=[dataset_1_train_repeat, dataset_2_train_repeat, dataset_3_train_repeat, dataset_4_train_repeat])

train_dataloader = dict(
    batch_size=256,
    dataset=dataset_concat
)

# Val dataloaders
dataset_1_val = dict(
    type=dataset_type,
    data_root='/data/its/vehicle_cls/vp3_202307_crop/',
    metainfo=metainfo,
    ann_file='annotations/val.txt',
    data_prefix='',
    with_label=True,
    pipeline=test_pipeline
)

dataset_2_val = dict(
    type=dataset_type,
    data_root='/data/its/vehicle_cls/202307_crop_ttp/',
    metainfo=metainfo,
    ann_file='annotations/val.txt',
    data_prefix='images/',
    with_label=True,
    pipeline=test_pipeline
)

dataset_3_val = dict(
    type=dataset_type,
    data_root='/data/its/vehicle_cls/vehicle_v3',
    metainfo=metainfo,
    ann_file='annotations/test.txt',
    data_prefix='images/',
    with_label=True,
    pipeline=test_pipeline
)

dataset_4_val = dict(
    type=dataset_type,
    data_root='/data/its/vehicle_cls/vehicle_v4',
    metainfo=metainfo,
    ann_file='annotations/test.txt',
    data_prefix='images/',
    with_label=True,
    pipeline=test_pipeline
)
# Apply concat dataset to val
dataset_val = dict(
    type='ConcatDataset',
    _delete_=True,
    datasets=[dataset_1_val, dataset_2_val,dataset_3_val, dataset_4_val])

val_dataloader = dict(
    batch_size=64,
    dataset=dataset_val
)

# accuracy
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# OR Precision, Recall, F1-score
val_evaluator = dict(
    _delete_=True,
    type='SingleLabelMetric',
    average=None, # Print out classwise
    # average='macro' # average
)

test_dataloader = val_dataloader
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=2)
default_hooks = dict(checkpoint=dict(type='CheckpointHook',
                     interval=10, max_keep_ckpts=2, save_best='auto'))

# fp16 training help you x2 batch size
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale=512.0)
