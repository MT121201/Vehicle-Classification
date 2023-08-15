##############################################################################################################
# Usage: python pseudo_label.py config_file \
#                                checkpoint_file \
#                                --img image_root_dir \
#                                --ann pseudo_annotation_file \
#                                --out_img final_folder \
#                                --out_ann annotations_output \
#                                --thresh threshold_predict_score \
#                                --CVAT path_to_image_in_CVAT
# This file will predict base on pseudo annotation file, if predict score > threshold and predict class same with its class in input annotation,
# move that image to final folder and write its annotation to output annotation file
# If CVAT is not None, correct the path in annotation file, for upload to CVAT
##############################################################################################################


import os
from mmpretrain import ImageClassificationInferencer
import shutil
from tqdm import tqdm
import argparse
import torch

def pseudo_label(inferencer, image_root_dir, pseudo_annotation_file, final_folder, annotations_output, thresh=0.8):
    
    # open pseudo annotation file
    with open(pseudo_annotation_file, 'r') as f:
        lines = f.readlines()

    # Create a list to store image paths and class indices
    annotations = []

    # Process each image in the folder and count
    processed_count = 0
    correct_count = 0
    process_bar = tqdm(total=len(lines))
    for line in lines:
        image_path, retrie_class = line.strip().split()
        image_name = os.path.basename(image_path)
        image_path = os.path.join(image_root_dir, image_name)
        # Check if the image exists
        if os.path.exists(image_path):
            predict = inferencer(image_path)
            class_score = predict[0]['pred_score']
            if class_score >thresh:
                class_predict = predict[0]['pred_label']
                # if class_predict same with retrie_class
                if class_predict == int(retrie_class):
                    
                    #Move image to final folder
                    new_image_name = f"{retrie_class}_{image_name}"
                    move_path = os.path.join(final_folder, new_image_name)
                    shutil.move(image_path, move_path)
                    annotations.append(move_path + ' ' + retrie_class)
                    correct_count += 1
            processed_count += 1
        else:
            print(f"Image {image_path} not found")
        process_bar.update()
    # Write image paths and class indices to a text file
    output_txt_file = annotations_output
    with open(output_txt_file, 'a') as f:
        for annotation in annotations:
            f.write(annotation + '\n')

    print("Annotations have been written to:", output_txt_file)
    print("Total images:", processed_count)
    print("Total images correct:", correct_count)
    process_bar.close()

def correct_ann(pseudo_annotation_file, CVAT):
    with open(pseudo_annotation_file, 'r') as f:
        lines = f.readlines()
    with open(pseudo_annotation_file, 'w') as f:
        for line in lines:
            image_path, class_index = line.strip().split()
            image_name = image_path.split('/')[-1]
            if CVAT is not None:
                image_name = os.path.join(CVAT, image_name)
            f.write(image_name + ' ' + class_index + '\n')

def parse_args():
    parser = argparse.ArgumentParser(description='Check anntation file, if similar predict, move that image and annotation to output folder')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='pretrained model file path')
    parser.add_argument('--img', help='image root directory')
    parser.add_argument('--ann', help='pseudo annotation file')
    parser.add_argument('--out_img', default= './cache/predict_images/',help='folder store similar images')
    parser.add_argument('--out_ann', default='./cache/prepredict_ann.txt', help='txt file store similar annotations')
    parser.add_argument('--thresh', default=0.8, type=float, help='threshold predict score')
    parser.add_argument('--CVAT', type=str, default=None, help='Path to image in CVAT, if not None, correct the path in pseudo_annotation.txt')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config_file = args.config
    checkpoint_file = args.checkpoint
    image_root_dir = args.img
    if not os.path.exists(image_root_dir):
        raise ValueError("Image root directory not found")
    pseudo_annotation_file = args.ann
    if not os.path.exists(pseudo_annotation_file):
        raise ValueError("Pseudo annotation file not found")
    final_folder = args.out_img
    annotations_output = args.out_ann
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)
    if not os.path.exists(annotations_output):
        open(annotations_output, 'w').close()

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        raise ValueError("Cuda is not available")
    inferencer = ImageClassificationInferencer(config_file, checkpoint_file, device)
    pseudo_label(inferencer, image_root_dir, pseudo_annotation_file, final_folder, annotations_output, args.thresh)
    if args.CVAT is not None:
        print("Detect CVAT path, correct pseudo annotation file")
    print("Check output annotation file")
    correct_ann(annotations_output, args.CVAT)
    print("Done")
if __name__ == '__main__':
    main() 