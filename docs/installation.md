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
pip install --no-cache-dir -U albumentations --no-binary qudida,albumentations click fastdup
pip install git+https://github.com/thuyngch/cvut

# install mmengine, mmcv, mmpretrain
pip install -U openmim
mim install "mmengine>=0.6.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmpretrain==1.0.0"
mim install mmdet==3.1.0

# install this repo
python setup.py develop
```
- TensorRT
```bash
pip install tensorrt
```
-   ONNXRuntime (GPU)
```bash
pip install onnxruntime-gpu==1.15.1
```

- If you intend to employ FastReID for Image Retrieval, kindly proceed to install it using the [Official FastReID repo](https://github.com/JDAI-CV/fast-reid/tree/master)
