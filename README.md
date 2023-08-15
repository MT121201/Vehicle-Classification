# Installation
- Create conda env
```bash
conda create -n ${user}_tnivc  python=3.9 -y
conda activate ${user}_tnivc
```
- Install libs:
```bash
# install pytorch
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# install other packages
pip install scipy tqdm future tensorboard
pip install timm==0.4.12 adabelief_pytorch motmetrics imagecorruptions
pip install git+https://github.com/jonbarron/robust_loss_pytorch
pip install --no-cache-dir -U albumentations --no-binary qudida,albumentations click
pip install git+https://github.com/thuyngch/cvut

# install mmengine, mmcv, mmpretrain
pip install -U openmim
mim install "mmengine>=0.6.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmpretrain==1.0.0"

# install this repo
python setup.py develop
```
- TensorRT
```bash
pip install tensorrt
```
-   ONNXRuntime (GPU)
```bash
pip install onnxruntime-gpu
```

# Utilization
### Train
```bash
export CUDA_VISIBLE_DEVICES=0
echo "train classification model"

WORKDIR="experiments/mobilenet-v3-small_8xb128_vc"

# config file
CFG="configs/tnivc/mobilenet-v3-small_8xb128_vc.py"

# train model
mim train mmpretrain $CFG --work-dir $WORKDIR
```

### Test
```bash
export CUDA_VISIBLE_DEVICES=0

# your config file
CFG="configs/tnivc/mobilenet-v3-small_8xb128_vc.py"

# the checkpoint
CHECKPOINT="experiments/mobilenet-v3-small_8xb128_vc/<YOUR_CKPT.pth>"

# save images dir
WORKDIR="experiments/mobilenet-v3-small_8xb128_vc/"
SHOWDIR=visualize

# run test
mim test mmpretrain $CFG --checkpoint $CHECKPOINT --work-dir $WORKDIR --show-dir $SHOWDIR
```

### Visualize trained dataset
```bash
python tools/visualizer/test_dataset.py \
    --config_file configs/tnivc/repvgg_vc.py \
    --save_img_dir ./cache/debugdata
```

### Pseudo annotation
```
 Gallery images folder structure:
    path
    |--- class_1
    |    |--- img_1
    |    |--- img_2
    |    |--- ...
    |--- class_2
    |---...

Query images folder structure:
    path
    |--- img_1
    |--- img_2
    |--- ...

synsets.txt file with classes_name in each line
|-class 1
|-class 2
|...
```
```bash
conda activate triet_fastreid
python retrieval/pseudo_ann/retrieval_onnx.py \
--query: path to Query folder, default = None
--gallery: path to Gallery folder, default = /data/its/vehicle_cls/image_retrieval
--synsets: path to synsets.txt file, default = /data/its/vehicle_cls/image_retrieval/synsets.txt
--CVAT: correct CVAT path, default = None 
--out: txt output annotation file, default = ./cache/annotation.txt
```
### Dataset tools
- Split dataset
```bash
python tools/dataset_tools/devide_dataset.py \
        --input: orginal annotations file \
        --thresh: split threshold, default is 0.85 \
        --out: output directory, default is in the same directory with input file
```
- Visualize class in dataset
```bash
python tools/dataset_tools/visualize_class_images.py
        annotation path \
        image root path \
        --c :class index want to visualize \
        --n :number images want to visualize, default = 50 \
        --o :output save image name, default = Visualize class index .jpg \
        --show :if show flag given, show image before saving
```