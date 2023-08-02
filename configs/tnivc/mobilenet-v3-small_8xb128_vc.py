# base model path
_base_ = '../mmpretrain/mobilenet_v3/mobilenet-v3-small_8xb128_in1k.py'
# classes
class_name = ('xe sedan', 'xe SUV', 'xe ban tai', 'xe ba gac', 'xe tai nho', 
                 'xe tai lon', 'xe container', 'xe may xuc, may cau, xe lu, xe cho be tong', 
                 'xe 16 cho', 'xe 29-32 cho', 'xe 52 cho' )
num_classes = len(class_name)
metainfo = dict(classes=class_name)

# Load pretrained model from github
load_from = "https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-small_8xb128_in1k_20221114-bd1bfcde.pth"

# Change num_classes in model head
model = dict(
    head=dict(num_classes=num_classes))

# Change num_classes in train data preprocessor
data_preprocessor = dict(
    num_classes=num_classes,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

# Move some train_pipeline to here
bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
    dict(type='GaussianBlur', prob=0.5, radius=2 ),
    dict(type ='Lighting', eigval=[0.2175, 0.0188, 0.0045], eigvec=[[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]],),
    dict(type='Rotate', angle= 20 ,prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomGrayscale', prob=0.2, keep_channels=True),
    dict(
        type='RandomErasing',
        erase_prob=0.2,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='RandomResize', scale=(224,224), ratio_range=(0.8,1.2)),
    dict(type='RandomCrop', crop_size=224, padding=(3,3,3,3), pad_if_needed=True),
    dict(type='PackInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]
# This dataset will repeat
dataset_A=dict(
    type='ImageNet',
    data_root='/data/its/vehicle_cls/vp3_202307_crop/',  
    metainfo=metainfo,
    ann_file='annotations/annotations.txt',
    data_prefix='',
    with_label=True,
    pipeline=train_pipeline
    )

# This dataset will keep remain
dataset_B=dict(
    type='ImageNet',
    data_root='/data/its/vehicle_cls/202307_crop_ttp/',  
    metainfo=metainfo,
    ann_file='annotations/annotations.txt',
    split='train',
    data_prefix='images/',
    with_label=True,
    pipeline=train_pipeline
    )

# Repeat then concat dataset
# dataset_A = dict(
#     type='RepeatDataset',
#     times=5,
#     dataset=dataset_to_repeat)

dataset_train = dict(
    type='ConcatDataset',
    _delete_=True,
    datasets=[dataset_A, dataset_B])

# Apply concat dataset to train dataloader
train_dataloader = dict(
    batch_size=128,
    dataset=dataset_train
)

# Val dataloaders use the normal dataset
val_dataloader = dict(
    batch_size=128,
    dataset=dict(
        type='ImageNet',
        data_root='/data/its/vehicle_cls/202307_crop_ttp/',  
        metainfo=metainfo,
        ann_file='annotations/annotations.txt',
        data_prefix='images/',
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader

train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=10)

optim_wrapper = dict(
    optimizer=dict(
        type='RMSprop',
        lr=1e-3,
        alpha=0.9,
        momentum=0.9,
        eps=0.0316,
        weight_decay=1e-05))

default_hooks = dict(checkpoint=dict(type='CheckpointHook',interval=10, max_keep_ckpts=2, save_best='auto'))
