import os
import argparse
from mmpretrain import ImageClassificationInferencer
from tqdm import tqdm
import torch


def inference(image_folder, save, inferencer, threshold=0.01):
    # list of image in predict folder
    images = os.listdir(image_folder)
    # Process each image in the folder and count
    processed_count = 0
    process_bar = tqdm(total=len(images))
    with open(save, 'w') as f:
        for image in images:
            image_path = os.path.join(image_folder, image)
            predict = inferencer(image_path)
            predict_scores = predict[0]['pred_score']
            if predict_scores >threshold:
                class_index = predict[0]['pred_label']
                f.write(f"{image} {class_index}\n")         
            else:
                f.write(f"{image} \n")
            processed_count += 1
            process_bar.update()

    print("Total images:", processed_count)
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
    parser = argparse.ArgumentParser(description='Using pretrain model to pseudo annotation')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='pretrained model file path')
    parser.add_argument('--img', help='image root directory')
    parser.add_argument('--out', default='./cache/predict/', help='output directory')
    parser.add_argument('--thresh', default=0.01, type=float, help='threshold predict score')
    parser.add_argument('--CVAT', default=None, help='If choose, prepare pseudo label for CVAT')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config_file = args.config
    checkpoint_file = args.checkpoint
    image_folder = args.img

    save = os.path.join(args.out, "pretrain_pseudo_annotation.txt")

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        raise ValueError("No GPU found")
    inferencer = ImageClassificationInferencer(config_file, checkpoint_file, device)
    inference(image_folder, save, inferencer, args.thresh)

    if args.CVAT is not None:
        print('Detect flag CVAT, now prepare pseudo label for CVAT')
        correct_ann(save, args.CVAT)
        print('Done prepare pseudo label for CVAT, check in ./cache/class_CVAT.txt but please check the image path because CVAT maybe using different path')
if __name__ == '__main__':
    main()