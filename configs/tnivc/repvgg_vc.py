_base_  = "../mmpretrain/repvgg/repvgg-A0_8xb32_in1k.py"

# classes
class_name = ('xe sedan', 'xe SUV', 'xe ban tai', 'xe ba gac', 'xe tai nho', 
                 'xe tai lon', 'xe container', 'xe may xuc, may cau, xe lu, xe cho be tong', 
                 'xe 16 cho', 'xe 29-32 cho', 'xe 52 cho' )
num_classes = len(class_name)
metainfo = dict(classes=class_name)

# Load pretrained model from github
load_from = "https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_8xb32_in1k_20221213-60ae8e23.pth"

# Change head numclasses and loss weight
model = dict(
    head=dict(num_classes=num_classes,
              loss=dict(type='CrossEntropyLoss', loss_weight=1.0,
                        #sedan,suv,bantai,bagac,tainho,tailon,container,mayxuc,16cho,29cho,52cho
                        class_weight=[0.82, 0.34, 1.0, 1.0, 1.0, 0.34, 1.0, 1.0, 1.0, 1.0, 1.0])))

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


# Train dataloaders
dataset_A_train=dict(
    type='ImageNet',
    data_root='/data/its/vehicle_cls/vp3_202307_crop/',  
    metainfo=metainfo,
    ann_file='annotations/train.txt',
    data_prefix='',
    with_label=True,
    pipeline=train_pipeline
    )

dataset_B_train=dict(
    type='ImageNet',
    data_root='/data/its/vehicle_cls/202307_crop_ttp/',  
    metainfo=metainfo,
    ann_file='annotations/train.txt',
    # split='train',
    data_prefix='images/',
    with_label=True,
    pipeline=train_pipeline
    )

# Apply class balanced sampler
dataset_balance_A = dict(
    type='ClassBalancedDataset',
    # _delete_=True,
    dataset=dataset_A_train,
    oversample_thr=0.1)

dataset_balance_B = dict(
    type='ClassBalancedDataset',
    # _delete_=True,
    dataset=dataset_B_train,
    oversample_thr=0.1)

# Concat dataset
dataset_concat = dict(
    type='ConcatDataset',
    _delete_=True,
    datasets=[dataset_balance_A, dataset_balance_B])

train_dataloader = dict(
    batch_size=64,
    dataset=dataset_concat
)

# Val dataloaders 
dataset_A_val=dict(
    type='ImageNet',
    data_root='/data/its/vehicle_cls/vp3_202307_crop/',  
    metainfo=metainfo,
    ann_file='annotations/val.txt',
    data_prefix='',
    with_label=True,
    pipeline=test_pipeline
    )

dataset_B_val=dict(
    type='ImageNet',
    data_root='/data/its/vehicle_cls/202307_crop_ttp/',  
    metainfo=metainfo,
    ann_file='annotations/val.txt',
    # split='train',
    data_prefix='images/',
    with_label=True,
    pipeline=test_pipeline
    )

dataset_val = dict(
    type='ConcatDataset',
    _delete_=True,
    datasets=[dataset_A_val, dataset_B_val])
# Apply concat dataset to val
val_dataloader = dict(
    batch_size=64,
    dataset=dataset_val
)

val_evaluator = dict(type='Accuracy', topk=(1, 5))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=2)
default_hooks = dict(checkpoint=dict(type='CheckpointHook',interval=10, max_keep_ckpts=2, save_best='auto'))