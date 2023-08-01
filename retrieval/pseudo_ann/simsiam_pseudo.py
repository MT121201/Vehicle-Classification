import os
import torch
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
from mmpretrain import get_model

def load_img_from_dir(root_gallery):
    """Load image from directory, directory structure:
    path
    |--- class_1
    |    |--- img_1
    |    |--- img_2
    |    |--- ...
    |--- class_2
    |---...
    Args:
        path: path to image directory
    Return:
        list of image path
        Dict of class index and image path
    """
    image_list = []
    for class_idx, class_name in enumerate(os.listdir(root_gallery)):
        class_path = os.path.join(root_gallery, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image_list.append(img_path)
    return image_list

def get_list_query_image_path(query_path):
    """Get list of image path from a folder
    Args: path to folder
    Return: list of image path
    """
    list_image = []
    for image in os.listdir(query_path):
        if image.endswith('.jpg'):
            list_image.append(os.path.join(query_path, image))
        if image.endswith('.png'):
            list_image.append(os.path.join(query_path, image))
    return list_image

def get_class_name(synsets_path):
    """Get class name from synsets.txt file
    Args:
        synsets_path: path to synsets.txt file
    Return:
        list of class name
    """
    class_name = []
    with open(synsets_path, 'r') as f:
        for line in f:
            class_name.append(line.strip())
    return class_name

# Preprocessing for images
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(img).unsqueeze(0)

# Function to extract features from an image
def extract_features(model, image_tensor):
    feats = model.extract_feat(image_tensor.to('cuda'))
    return feats[0]

# Function to compute cosine similarity between query and gallery features
def cosine_similarity(feature_map1, feature_map2):
    similarity_feats = F.cosine_similarity(feature_map1.view(1, 2048, -1), feature_map2.view(1, 2048, -1)).detach().cpu().numpy()[0]
    mean_similarity = np.mean(similarity_feats)
    return mean_similarity

def generate_pseudo_labels(query_images_list, gallery_images_list, model):
    gallery_features_dict = {}
    for gallery_path in gallery_images_list:
        gallery_image = load_and_preprocess_image(gallery_path)
        gallery_feature_map = extract_features(model, gallery_image)
        gallery_features_dict[gallery_path] = gallery_feature_map

    query_labels = {}
    for query_path in query_images_list:
        query_image = load_and_preprocess_image(query_path)
        query_feature_map = extract_features(model, query_image)
        
        best_match = None
        best_similarity = -1
        for gallery_filename, gallery_features in gallery_features_dict.items():
            similarity = cosine_similarity(query_feature_map, gallery_features)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = gallery_filename
        query_labels[query_path] = os.path.splitext(best_match)[0]
    return query_labels

def get_predict_class(path_of_highest_similar_image, class_names):
    """Get class of predicted image from path of highest score similar image
    Args:
        path_of_highest_similar_image: path of highest score similar image
        class_names: list of class name
    Return:
        class index of predicted image
    """
    class_name = path_of_highest_similar_image.split('/')[-2]
    class_idx = class_names.index(class_name)
    return class_idx

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

def write_pseudo_annotation(pseudo_labels, class_names, CVAT_image_path=None):
    """Write pseudo annotation to file in ImageNet format
    Args:
        pseudo_labels: Dict, Image: query image name, Pseudo Label: most similar image name
        class_names: list of class name
        CVAT_image_path: path to image in CVAT, if not None, correct the path in pseudo_annotation.txt
    Return:
        None, but write the pseudo_annotation.txt file
    """
    with open('pseudo_annotation.txt', 'w') as f:
        for query in pseudo_labels:
            query_path = query
            gallery_path = pseudo_labels[query]
            # Correct path like CVAT, if not None
            if CVAT_image_path != None:
                query_path = correct_image_path(CVAT_image_path, query_path)
            # Get class index
            class_idx = get_predict_class(gallery_path, class_names)
            annotation = str(query_path) + ' ' + str(class_idx)
            f.write(annotation)
            f.write('\n')
def main():
    # Define the path to query and gallery images
    # Query images folder structure:
    # path
    # |--- img_1
    # |--- img_2
    # |--- ...
    query_images_folder = "/data/its/vehicle_cls/vp3_202307_crop"
    # Gallery images folder structure:
    # path
    # |--- class_1
    # |    |--- img_1
    # |    |--- img_2
    # |    |--- ...
    # |--- class_2
    # |---...
    gallery_images_folder = "retrieval/gallery"
    # Path to synsets.txt file; the txt file with class name in each line:
    # class1
    # class2
    # ...
    class_synsets = '/home/tni/Workspace/triet/Vehicle-Classification/retrieval/pseudo_ann/synsets.txt'

    # Correct path in CVAT, if None, return the real path
    # Sometime the path in repo is not same as path in CVAT, so we need to correct the path
    # SET NONE IF NOT USE
    CVAT_image_path = 'its/vehicle_cls/vp3_202307_crop/' 

    # Load the pre-trained model and move it to CUDA
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        raise ValueError('No GPU found!')
    model = get_model('simsiam_resnet50_8xb32-coslr-200e_in1k',device=device, pretrained=True)
    model.eval()

    # Get list of query and gallery images
    query_images_list = get_list_query_image_path(query_images_folder)
    gallery_images_list = load_img_from_dir(gallery_images_folder)
    
    # Get class name from synsets.txt file
    class_names = get_class_name(class_synsets)

    # Generate pseudo labels
    pseudo_labels = generate_pseudo_labels(query_images_list, gallery_images_list, model)

    # Write pseudo annotation, if CVAT_image_path != None, correct the path in pseudo_annotation.txt
    if CVAT_image_path != None:
        print('Detect CVAT image path, now correct the query path in pseudo_annotation.txt')
    write_pseudo_annotation(pseudo_labels, class_names, CVAT_image_path)
    print('Done, check pseudo_annotation.txt')

if __name__ == "__main__":
    main()