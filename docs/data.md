### Detect vehicle from video and crop it
```bash
python tools/dataset_tools/video_predict_crop.py \
--video_dir directory contain videos to crop \
--save folder to save cropped images
```
### Fastdup
Usage: Clean image after crop from video
```bash
python tools/dataset_tools/clean_crop_images.py \
        --input image dir \
        --blur blur threshold, default =70
        --brightness white brightness threshold, default = 220
        --darkness darkness threshold, default = 50
        --outliers outlier threshold, default = 0.68
```

### Pseudo annotation by Retrieval
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
python retrieval/pseudo_ann/retrieval_onnx.py \
--query: path to Query folder, default = None
--gallery: path to Gallery folder, default = /data/its/vehicle_cls/gallery_retrieval
--synsets: path to synsets.txt file, default = /data/its/vehicle_cls/synsets.txt
--CVAT: correct CVAT path, default = None 
--out: txt output annotation file, default = ./cache/annotation.txt
```
### Dataset tools
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
#### Demo
- Crop Video
```bash
python tools/dataset_tools/video_predict_crop.py \
--video_dir /data/its/vehicle_cls/demo/video \
--save /data/its/vehicle_cls/demo/crop_from_video
```
- Clean after crop
```bash
python tools/dataset_tools/clean_crop_images.py \
        --input /data/its/vehicle_cls/demo/crop_from_video
```
- Retrieval Pseudo Annotation
```bash
python retrieval/pseudo_ann/retrieval_onnx.py \
--query /data/its/vehicle_cls/demo/crop_from_video \
--CVAT its/vehicle_cls/demo/crop_from_video
--out /data/its/vehicle_cls/demo/annotations
```
- Post CVAT processing
```bash
python tools/dataset_tools/post_processing_ann.py \
        /data/its/vehicle_cls/demo/annotations/pseudo_annotation.txt \ 
        /data/its/vehicle_cls/demo/crop_from_video \ 
        --ds_img /data/its/vehicle_cls/demo/images \
        --ds_ann /data/its/vehicle_cls/demo/annotations 
```