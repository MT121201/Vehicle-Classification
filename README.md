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
# download the TensorRT tar file from NVIDIA here use 8.6.1.6 cuda 11.x to deploy dir
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
# extract it to the current directory
tar -zxvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
# install TensorRT with cp3x
pip install TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp39-none-linux_x86_64.whl
# Change path for install PyCuda
export CPATH=$CPATH:/usr/local/cuda-11.7/targets/x86_64-linux/include
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-11.7/targets/x86_64-linux/lib
pip install pycuda
#
# Download CuNN from NVIDIA and move download file to current directory before next line
#
tar xf cudnn-linux-x86_64-8.9.0.131_cuda11-archive.tar.xz
export CUDNN_DIR=$(pwd)/cuda
export LD_LIBRARY_PATH=$CUDNN_DIR/lib64:$LD_LIBRARY_PATH
```
-   ONNXRuntime (GPU)
```bash
# you can install one to install according whether you need gpu inference
# onnxruntime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.0/onnxruntime-linux-x64-1.15.0.tgz
tar -zxvf onnxruntime-linux-x64-1.15.0.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.15.0
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH

# onnxruntime-gpu
pip install onnxruntime-gpu==1.15.0
wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.0/onnxruntime-linux-x64-gpu-1.15.0.tgz
tar -zxvf onnxruntime-linux-x64-gpu-1.15.0.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-gpu-1.15.0
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
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
    --config_file configs/tnivc/mobilenet-v3-small_8xb128_vc.py \
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

ONNX:
python retrieval/pseudo_ann/retrieval_onnx.py \
--query: path to Query folder, default = None
--gallery: path to Gallery folder, default = /data/its/vehicle_cls/image_retrieval
--synsets: path to synsets.txt file, default = /data/its/vehicle_cls/image_retrieval/synsets.txt
--CVAT: correct CVAT path, default = None 
--out: txt output annotation file, default = ./cache/annotation.txt

Simian:
python retrieval/pseudo_ann/simsiam_pseudo.py
-- //same as ONNX//
```
- Split dataset
```bash
python tools/check_data/devide_dataset.py \
        --input orginal annotations file \
        --thresh split threshold, default is 0.85 \
        --out output directory, default is in the same directory with input file
```
- Calculate loss weight for each class
```bash
# Change num_class and path to annotation files in tools/dataset_tools/calculate_loss_weight.py 
python tools/dataset_tools/calculate_loss_weight.py 
```