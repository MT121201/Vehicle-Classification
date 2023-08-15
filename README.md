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
- Adding to retrieval Galley folder

Usage: Use this tool to make or adding image from dataset to Gallery folder prepare for retrieval
```bash
python tools/dataset_tools/make_gallery.py \
        --synset: synset txt file contain class name \
        --img: path to image folder \
        --ann: path to annotation file using \
        --gal: path to Gallery folder \
        --n: number of images per class adding to gallery
```
- Pretrain model inference on pseudo annotation

When we have to pseudo annotation from retrieval, we may use this tool to check if predict of retrieval is similar with pretrain predict 
if True, move that image to output folder and write its label to output annotation.txt
```bash
python tools/dataset_tools/infer_on_pseu_ann.py \
        config_file \
        checkpoint_path \
        --img image folder \ 
        --ann pseudo annotation txt \
        --out_img output predict image folder path, default = "./cache/predict_images" \
        --out_ann output predict annotation txt, default = "./cache/predict_ann.txt" \
        --thresh threshold of predict scores, default = 0.8 \
        --CVAT path of images in CVAT, if given, path in output annotation file will correct with this path, default is None
```
- Predict specific class tool

When we get image folder with is huge and we just need find out just our finding class's images, use this tool will pick the images 
which predict score of given class above threshold to output folder/predict_{class_index}/image.jpg, for speed up CVAT checking process
```bash
python tools/dataset_tools/predict_specific_class.py \
        config_file \
        checkpoint_path \
        --img image folder to predict \
        --c specific class to get predict, can input several classes, eg: --c 3,4,5 \
        --out output directory, defaults = "./cache/predict"
        --thresh threshold of predict scores, default = 0.01
        --remove if set will delete origin images in input folder if it exist in output folder
        --CVAT if set will create annotation file of pseudo labels for each output folder, class will same in this file
```
- Post CVAT processing

Usage: After checking by CVAT, use this to remove None class, and move image, label to dataset
```bash
python tools/dataset_tools/post_processing_ann.py \
        CVAT output txt annotation path \ 
        image folder path, which using in CVAT \ 
        --ds_img dataset image folder path,  default='./cache/dataset/images/'
        --ds_ann annotation folder path, default='./cache/dataset/annotations/'
        --delete if set, delete image have None class in CVAT output
```
-Fastdup

Usage: Clean image after crop from video
```bash
python tools/dataset_tools/fastdup.py \
        --input image dir \
        --blur blur threshold, default =70
        --brightness white brightness threshold, default = 220
        --darkness darkness threshold, default = 50
        --outliers outlier threshold, default = 0.68
```