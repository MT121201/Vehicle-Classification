## Instruction video
Before proceeding, it's important to watch [this video]() in order to gain an understanding of how to perform image retrieval, upload, labeling, and model training.

## Detect vehicle from video and crop it
```bash
python tools/dataset_tools/video_predict_crop.py \
--video_dir directory contain videos to crop \
--save folder to save cropped images
```
## Fastdup
- Clean image after crop from video
```bash
python tools/dataset_tools/clean_crop_images.py \
        --input image dir \
        --blur blur threshold, default =70
        --brightness white brightness threshold, default = 220
        --darkness darkness threshold, default = 50
        --outliers outlier threshold, default = 0.68
```

## Pseudo annotation
### Retrieval
#### 1. [Retrieval] - Folder struture
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

#### 2. [Retrieval] - Run Image Retrieval using the provided example below:

- We will use a demo video under `/data/its/vehicle_cls/demo/video` to illustrate the process.

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

- Adding to retrieval Galley folder
Usage: Use this tool to make or adding image from dataset to Gallery folder prepare for retrieval
```bash
python tools/dataset_tools/make_gallery.py \
        --synset /data/its/vehicle_cls/synsets.txt \
        --img /data/its/vehicle_cls/vehicle_v5/images \
        --ann /data/its/vehicle_cls/vehicle_v5/annotations/annotations.txt \
        --gal /data/its/vehicle_cls/gallery_retrieval --n 10
```

- Retrieval Pseudo Annotation
```bash
python retrieval/pseudo_ann/retrieval_onnx.py \
        --query /data/its/vehicle_cls/demo/crop_from_video \
        --gallery /data/its/vehicle_cls/gallery_retrieval \
        --CVAT its/vehicle_cls/demo/crop_from_video \
        --out /data/its/vehicle_cls/demo/annotations/pseudo_annotation.txt \
        --synsets /data/its/vehicle_cls/synsets.txt
```

- Upload the data to CVAT to label.

- Post CVAT processing: After labeling by CVAT, use this to remove None class, and move image, label to dataset
```bash
python tools/dataset_tools/post_processing_ann.py \
        /data/its/vehicle_cls/demo/annotations/pseudo_annotation.txt \ 
        /data/its/vehicle_cls/demo/crop_from_video \ 
        --ds_img /data/its/vehicle_cls/demo/images \
        --ds_ann /data/its/vehicle_cls/demo/annotations 
```

#### 3. Pseudo using trained classification model
- TODO: Add an instruction video and demo commands in this case

#### Useful tools
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
