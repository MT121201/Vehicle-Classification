import os
import shutil
import argparse

def delete_none_class_image(ann_path, cvat_img_path):
    count = 0
    with open(ann_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split(' ')
        if len(parts) == 2:
            pass
        elif len(parts) == 1:
            path = parts[0]
            name = os.path.basename(path)
            true_path = os.path.join(cvat_img_path, name)
            delete_image(true_path, count)
    print(f"Deleted {count} images")
def delete_image(image_path, count):
    if os.path.exists(image_path):
        os.remove(image_path)
        count += 1
    else:
        print(f"Error deleting: {image_path}")

def clean_none_class_in_ann_file(ann_path):
    lines_with_class = []
    with open(ann_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        elements = line.strip().split()
        if len(elements) >= 2:
            lines_with_class.append(line)
    sorted_lines = sorted(lines_with_class, key=lambda line: int(line.strip().split()[1]))
    with open(ann_path, 'w') as f:
        for line in sorted_lines:
            f.write(line)
    print('Done clean none class in annotation file')

def move_img(old_ann, cvat_img_path, img_dataset_folder, dataset_ann):
    count = 0
    ann_file = os.path.join(dataset_ann, 'annotations.txt')
    with open(old_ann, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split(' ')
        if len(parts) == 2:
            path, class_index = parts
            img_name = os.path.basename(path)
            true_img_path = os.path.join(cvat_img_path, img_name)
            shutil.move(true_img_path, os.path.join(img_dataset_folder, img_name))
            with open(ann_file, 'a') as f:
                f.write(f"{img_name} {class_index}\n")
            count += 1
    print(f"Moved {count} images")
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('cvat_ann', type=str, default=None, help='Path to annotation output from CVAT')
    parser.add_argument('cvat_img', type=str, default=None, help='Correct path of image using in CVAT')
    parser.add_argument('--ds_img', type=str, default='./cache/dataset/images/', help='Path to image dataset folder')
    parser.add_argument('--ds_ann', type=str, default='./cache/dataset/annotations/', help='Path to dataset annotation file')
    parser.add_argument('--delete', action='store_true', help='Delete image with none class after CVAT')
    args = parser.parse_args()
    return args

def main():
    args = parser()
    if args.cvat_ann is None or args.cvat_img is None:
        raise ValueError('Please provide correct path to annotation and image folder')
    if not os.path.exists(args.ds_img):
        os.makedirs(args.ds_img)
    if not os.path.exists(args.ds_ann):
        os.makedirs(args.ds_ann)
    if args.delete:
        print('Delete image with none class after CVAT')
        delete_none_class_image(args.cvat_ann, args.cvat_img)
    clean_none_class_in_ann_file(args.cvat_ann)
    move_img(args.cvat_ann, args.cvat_img, args.ds_img, args.ds_ann)
if __name__ == '__main__':
    main()
