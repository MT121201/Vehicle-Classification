import os
import shutil
import argparse
from mmpretrain import ImageClassificationInferencer
from tqdm import tqdm
import torch


def inference(folder_path, root_folder, inferencer, classes, threshold=0.01):
    # list of image in predict folder
    images = os.listdir(folder_path)
    # Process each image in the folder and count
    processed_count = 0
    class_count = {}
    process_bar = tqdm(total=len(images))
    for image in images:
        image_path = os.path.join(folder_path, image)
        # Check if the image exists
        if os.path.exists(image_path):
            predict = inferencer(image_path)
            predict_scores = predict[0]['pred_scores']
            for class_index in classes:
                class_score = predict_scores[int(class_index)]
                if class_score >threshold:
                    new_image_name = f"{class_index}_{image}"
                    class_predict_folder = os.path.join(root_folder, f"predict_{class_index}")
                    move_path = os.path.join(class_predict_folder, new_image_name)
                    shutil.copy(image_path, move_path)
                    if class_index not in class_count:
                        class_count[class_index] = 1
                    else:
                        class_count[class_index] += 1
            processed_count += 1
        else:
            print(f"Image {image_path} not found")
        process_bar.update()

    print("Total images:", processed_count)
    for class_index in classes:
        print(f"Class {class_index} has {class_count[class_index]} images")
    process_bar.close()

def remove_img_in_img_folder(root_img_folder, root_path, classes):
    for class_index in classes:
        class_predict_img_folder = os.path.join(root_path, f"predict_{class_index}")
        for img in os.listdir(class_predict_img_folder):
            img_new_name = img.split("/")[-1]
            img_name = img_new_name.split("_")[1:]
            img_old_name = "_".join(img_name)
            img_root_path = os.path.join(root_img_folder, img_old_name)
            if os.path.exists(img_root_path):
                os.remove(img_root_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Check anntation file, if similar predict, move that image and annotation to output folder')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='pretrained model file path')
    parser.add_argument('--img', help='image root directory')
    parser.add_argument('--out', default='./cache/predict/', help='output directory')
    parser.add_argument('--c', type=str, help='list of classes index to check')
    parser.add_argument('--thresh', default=0.01, type=float, help='threshold predict score')
    parser.add_argument('--remove', action='store_true', help='If choose, remove image in image folder after copy to predict folder')
    parser.add_argument('--CVAT', action='store_true', help='If choose, prepare pseudo label for CVAT')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config_file = args.config
    checkpoint_file = args.checkpoint
    image_folder = args.img
    classes = args.c.split(",")

    out_folder = os.path.dirname(args.out)
    for class_index in classes:
        class_predict_folder = os.path.join(out_folder, f"predict_{class_index}")
        os.makedirs(class_predict_folder, exist_ok=True)
        print(f"Predicting class {class_index} will be save to {class_predict_folder}")
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        raise ValueError("No GPU found")
    inferencer = ImageClassificationInferencer(config_file, checkpoint_file, device)
    inference(image_folder, out_folder, inferencer, classes, args.thresh)
    if args.remove:
        print('Detect flag remove, now remove root images in image folder if it exists in predict folder')
        remove_img_in_img_folder(image_folder, out_folder, classes)

    if args.CVAT:
        print('Detect flag CVAT, now prepare pseudo label for CVAT')
        out_folder_full_path =os.path.abspath(out_folder)

        for class_index in classes:
            class_predict_folder = os.path.join(out_folder_full_path, f"predict_{class_index}")
            with open (f"./cache/{class_index}_CVAT.txt", 'w') as f:
                for img in os.listdir(class_predict_folder):
                    img_path_nor = os.path.join(class_predict_folder, img)
                    img_path_cvat = img_path_nor.split("/")[1:]
                    img_path_cvat = "/".join(img_path_cvat)
                    f.write(f"{img_path_cvat} {class_index}\n")
        print('Done prepare pseudo label for CVAT, check in ./cache/class_CVAT.txt but please check the image path because CVAT maybe using different path')
if __name__ == '__main__':
    main()