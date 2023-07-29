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
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='AutoAugment',
        policies='imagenet',
        hparams=dict(pad_val=[round(x) for x in bgr_mean])),
    dict(
        type='RandomErasing',
        erase_prob=0.2,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackInputs'),
]
### To use dataset wrapper, have to change in configs/mmpretrain/_base_/datasets/imagenet_bs128_mbv3.py
### Line 44: dataset=dict(..) <- in train dataloader,we change this line to:
### Line 44: dataset=dict(
###             type =dataset_type,
###             datasets=dict(...)) ### Notice adjust indent, datasets <- "s"
### Eg: 
# train_dataloader = dict(
#     batch_size=128,
#     num_workers=5,
#     dataset=dict(
#         type=dataset_type,
#         datasets=dict(type=dataset_type,
#             data_root='data/imagenet',
#             pipeline=train_pipeline)),
#     sampler=dict(type='DefaultSampler', shuffle=True),
# )
### Without change, the config will load from base and ignore the dataset wrapper -> lead to error

# Dataset 
### Use CustomDataset type dont have split, instead use _delete_=True will lead to unknow error of dataloaders,
### please comment the "split" line in Train and Val dataloader in configs/mmpretrain/_base_/datasets/imagenet_bs128_mbv3.py
dataset_A = dict(
    type= 'RepeatDataset',
    times = 5,
    dataset=dict(
        type='CustomDataset',
        data_root='/data/its/vehicle_cls/vp3_202307_crop/',  
            metainfo=metainfo,
            ann_file='annotations/annotations.txt',
            data_prefix='',
            with_label=True,
            pipeline=train_pipeline
        )
)

dataset_B = dict(
    type= 'RepeatDataset',
    times = 2,
    dataset=dict(
        type='CustomDataset',
        data_root='/data/its/vehicle_cls/vp3_202307_crop/',  
            # _delete_=True,
            metainfo=metainfo,
            ann_file='annotations/annotations.txt',
            data_prefix='',
            with_label=True,
            pipeline=train_pipeline
        )
)


# Apply concat dataset to train dataloader
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
    type='ConcatDataset',
    datasets=[dataset_A, dataset_B])
)
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        type='CustomDataset',
        data_root='/data/its/vehicle_cls/vp3_202307_crop/',  
        # _delete_=True,
        metainfo=metainfo,
        ann_file='annotations/annotations.txt',
        data_prefix=''
    )
)
test_dataloader = val_dataloader

train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=10)

### learning rate default was 0.064, and don't see it scale with batch size, I change it to 1e-3
### With some model, when it scale it will print in log 
optim_wrapper = dict(
    optimizer=dict(
        type='RMSprop',
        lr=1e-3,
        alpha=0.9,
        momentum=0.9,
        eps=0.0316,
        weight_decay=1e-05))

default_hooks = dict(checkpoint=dict(type='CheckpointHook',interval=10, max_keep_ckpts=2, save_best='auto'))
