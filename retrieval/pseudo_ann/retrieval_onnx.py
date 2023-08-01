import os
import numpy as np
import onnxruntime as ort
from torchvision import transforms
from PIL import Image

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

# Load the ONNX model
onnx_model_path = "outputs/onnx_model/veriwild_vc.onnx"
sess = ort.InferenceSession(onnx_model_path)

# Preprocessing for images
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract features from an image
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).numpy()
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    features = sess.run([output_name], {input_name: image_tensor})[0]
    return features.squeeze()

def cosine_similarity(query_features, gallery_features):
    return np.dot(query_features, gallery_features) / (
        np.linalg.norm(query_features) * np.linalg.norm(gallery_features)
    )

def generate_pseudo_labels(query_images_folder, gallery_images_list):
    gallery_features_dict = {}
    query_labels = {}

    # Extract gallery features
    for image_path in gallery_images_list:
        gallery_features = extract_features(image_path)
        gallery_features_dict[image_path] = gallery_features
    
    # Extract query features and compute similarity
    for filename in os.listdir(query_images_folder):
        if filename.endswith('.jpg'):
            query_image_path = os.path.join(query_images_folder, filename)
            query_features = extract_features(query_image_path)

            best_match = None
            best_similarity = -1

            for gallery_filename, gallery_features in gallery_features_dict.items():
                similarity = cosine_similarity(query_features, gallery_features)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = gallery_filename

            query_labels[filename] = os.path.splitext(best_match)[0]
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
        None, but write the onnx_pseudo_annotation.txt file
    """
    with open('onnx_pseudo_annotation.txt', 'w') as f:
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
    gallery_images_folder = "/home/tni/Workspace/triet/Vehicle-Classification/retrieval/gallery"
    # Path to synsets.txt file; the txt file with class name in each line:
    # class1
    # class2
    # ...
    class_synsets = '/home/tni/Workspace/triet/Vehicle-Classification/retrieval/pseudo_ann/synsets.txt'
    
    # Correct path in CVAT, if None, return the real path
    # Sometime the path in repo is not same as path in CVAT, so we need to correct the path
    # SET NONE IF NOT USE
    CVAT_image_path = 'its/vehicle_cls/vp3_202307_crop/' ## set None if not use

    gallery_images_list = load_img_from_dir(gallery_images_folder)
    class_names = get_class_name(class_synsets)

    pseudo_labels = generate_pseudo_labels(query_images_folder, gallery_images_list)
    
    # Write pseudo annotation, if CVAT_image_path != None, correct the path in pseudo_annotation.txt
    if CVAT_image_path != None:
        print('Detect CVAT image path, now correct the query path in pseudo_annotation.txt')
    write_pseudo_annotation(pseudo_labels, class_names, CVAT_image_path)
    print('Done, check onnx_pseudo_annotation.txt')

if __name__ == "__main__":
    main()