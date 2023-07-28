### Usage: python pseudo_annotations.py
#############################################################################################
### Config in __main__ function: model_path, gallery_path, query_path, ann_path, CVAT_image_path
### model_path: path to model config (configs/mmpretrain/arcface/resnet50-arcface_8xb32_inshop.py)
### gallery_path: path to gallery image folder (Notice just get .jpg and .png file)
### query_path: path to query image folder (Notice just get .jpg and .png file)
### ann_path: path to Gallery annotation file (gallery_ann.txt), ImageNet format
### CVAT_image_path: path to image in CVAT, if not None, correct the path in output file (pseudo_annotation.txt)
#############################################################################################
### Output: pseudo_annotation.txt  
### Format: ImageNet
### Running pipeline:
###     1. Get list of gallery image path and query image path
###     2. Do inference query image with gallery image
###      - Using mmpretrain.ImageRetrievalInferencer
###      - Return list of query image path and the path of its predicted gallery image
###     3. Write pseudo annotation to file in ImageNet format
###      - If CVAT_image_path != None, correct the path in pseudo_annotation.txt
###      - Get class index of gallery image from annotation file, now it is annotation of this Query image
###      - Write to file in ImageNet format, each line is: query_image_path + class_index
#############################################################################################
from mmpretrain import ImageRetrievalInferencer
import os

def get_list_image(path_to_folder):
    """Get list of image path from a folder
    Args: path to folder
    Return: list of image path
    Note: only get .jpg and .png file
    """
    list_image = []
    for image in os.listdir(path_to_folder):
        if image.endswith('.jpg'):
            list_image.append(os.path.join(path_to_folder, image))
        if image.endswith('.png'):
            list_image.append(os.path.join(path_to_folder, image))
    return list_image

def inference_query_image(model_path, list_gallery, list_query):
    """Inference Query image with gallery image
    Args:
        model_path: path to model
        list_gallery: list of gallery image path
        list_query: list of query image path
    Return:
        query_path_n_gallery_path: list of query image path and the path of its predicted gallery image
    """
    # Create inferencer, load model, prototype can be |str | list | dict | DataLoader, BaseDataset| here we use list
    inferencer = ImageRetrievalInferencer(model_path, prototype=list_gallery)
    query_path_n_gallery_path = []
    for query in list_query:
        query_path = query
        predict = inferencer(query, topk=1)[0] # Get the highest one
        gallery_path = predict[0]['sample']['img_path'] # Get the image path of preidicted gallery image (prototype)
        query_path_n_gallery_path.append(str(query_path) + ' ' + str(gallery_path))
    return query_path_n_gallery_path
 
def get_gallery_idex_class(path_ann, path_gallery):
    """Get class index of gallery image from annotation file, 2nd element in line (ImageNet format)
    Args:
        path_ann: path to Gallery annotation file
        path_gallery: path to gallery image, get from inference_query_image
    Return:
        class_idx: class index of gallery image
    Error if not found path in annotation file
    """
    with open(path_ann, 'r') as f:
        for line in f:
            if line.split(' ')[0] == path_gallery:
                return line.split(' ')[1].replace('\n', '')
        raise ValueError('Not found path in annotation file')

def correct_image_path(real_path, query_img_path):
    """ For upload annotation to CVAT, sometimes the path is not correct (missing or extra some part of path)
        So check the path of image in CVAT and use this function to correct the path
        If the corect path is not found, return the real path
    Args:
        real_path: path to image in CVAT
        query_predict_path: query image path which is incorect in CVAT
    Return:
        correct_path: correct path of Query image in CVAT
    """
    img_name = query_img_path.split('/')[-1]
    correct_path = os.path.join(real_path, img_name)
    return correct_path

def write_pseudo_annotation(predict, ann_path, CVAT_image_path=None):
    """Write pseudo annotation to file in ImageNet format
    Args:
        predict: list of query image path and the path of its predicted gallery image
        ann_path: path to Gallery annotation file
        CVAT_image_path: path to image in CVAT, if not None, correct the path in pseudo_annotation.txt
    Return:
        None, but write the pseudo_annotation.txt file
    """
    with open('pseudo_annotation.txt', 'w') as f:
        for query in predict:
            query_path = query.split(' ')[0]
            gallery_path = query.split(' ')[1]
            # Correct path like CVAT, if not None
            if CVAT_image_path != None:
                query_path = correct_image_path(CVAT_image_path, query_path)
            # Get class index of gallery image from annotation file
            class_idx = get_gallery_idex_class(ann_path, gallery_path)
            annotation = str(query_path) + ' ' + str(class_idx)
            f.write(annotation)
            f.write('\n')

if __name__ == '__main__':
    # Path to model config, gallery folder, query folder, annotation file
    model_path = 'configs/mmpretrain/arcface/resnet50-arcface_8xb32_inshop.py'
    gallery_path = '/home/tni/Workspace/triet/Vehicle-Classification/test_imgs/gallery'
    query_path = '/data/its/vehicle_cls/vp3_202307_crop/'
    ann_path ='/home/tni/Workspace/triet/Vehicle-Classification/test_imgs/gallery/gallery_ann.txt'

    # Correct path in CVAT, if None, return the real path
    CVAT_image_path = 'its/vehicle_cls/vp3_202307_crop/' ## set None if not use

    # Get image path list
    list_gallery = get_list_image(gallery_path) 
    list_query = get_list_image(query_path)

    # Do inference
    predict = inference_query_image(model_path, list_gallery, list_query)
    
    # Write pseudo annotation, if CVAT_image_path != None, correct the path in pseudo_annotation.txt
    if CVAT_image_path != None:
        print('Detect CVAT image path, now correct the query path in pseudo_annotation.txt')
    write_pseudo_annotation(predict, ann_path, CVAT_image_path)
    print('Done, check pseudo_annotation.txt')
    

    