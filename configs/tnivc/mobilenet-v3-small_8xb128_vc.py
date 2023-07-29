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
    num_classes=num_classes)

### To use dataset wrapper, have to change in configs/mmpretrain/_base_/datasets/imagenet_bs128_mbv3.py
### Line 44: dataset=dict(..) <- in train dataloader,we change this line to:
### Line 44: dataset=dict(
###             dataset=dict(...)) ### Notice adjust indent
### Without change, the config will load from base and ignore the dataset wrapper -> lead to error

# Dataset 
### Use CustomDataset type dont have split, instead use _delete_=True will lead to unknow error of dataloaders,
### please comment the "split" line in Train and Val dataloader in configs/mmpretrain/_base_/datasets/imagenet_bs128_mbv3.py
dataset_A = dict(
    type= 'repeat_dataset',
    times = 5,
    dataset=dict(
        type='CustomDataset',
        data_root='/data/its/vehicle_cls/vp3_202307_crop/',  
            metainfo=metainfo,
            ann_file='annotations/annotations.txt',
            data_prefix='',
            with_label=True,
        )
)

dataset_B = dict(
    type= 'repeat_dataset',
    times = 2,
    dataset=dict(
        type='CustomDataset',
        data_root='/data/its/vehicle_cls/vp3_202307_crop/',  
            # _delete_=True,
            metainfo=metainfo,
            ann_file='annotations/annotations.txt',
            data_prefix='',
            with_label=True,
        )
)

data_train = dict(
    type='concat_dataset',
    datasets=[dataset_A, dataset_B]
)

# Apply concat dataset to train dataloader
train_dataloader = dict(
    batch_size=32,
    dataset=data_train
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
### If lr don't be scale automatically, set lr = 1e-3

default_hooks = dict(checkpoint=dict(type='CheckpointHook',interval=10, max_keep_ckpts=2, save_best='auto'))