import os
import shutil
import random
import argparse

# Create a function to move images to class folders.
def copy_image_to_class_folder(gallery_folder_path, image_path, class_label):

    
    class_folder_path = os.path.join(gallery_folder_path, str(class_label))
    os.makedirs(class_folder_path, exist_ok=True)
    
    shutil.copy(image_path, class_folder_path)

# Dictionary to keep track of image counts per class
class_image_counts = {}

def add_new_data(class_name, image_gallery_folder_path, txt_file_path, img_root, max_num_per_class=100):
# Open the text file and read it line by line
    imgname_index = []
    with open(txt_file_path, "r") as file:
        for line in file:
            # Split the line into image name and class index
            imgname_index.append(line)
            
    random.shuffle(imgname_index)
    for line in imgname_index:
        elements = line.split()
        # Check if the line contains at least two elements (image name and class index)
        if len(elements) >= 2:
            image_name = elements[0]
            class_index = elements[1]
            class_label = class_name[int(class_index)]
            # Build the path to the image file
            image_path = os.path.join(img_root, image_name)

            if os.path.exists(image_path):
                # Check if the class image count is within the limit
                if class_index not in class_image_counts:
                    class_image_counts[class_index] = 0
                
                if class_image_counts[class_index] < max_num_per_class:
                # Move the image to the appropriate class folder
                    copy_image_to_class_folder(image_gallery_folder_path, image_path, class_label)
                    class_image_counts[class_index] += 1
            else:
                print(f"Image {image_path} not found")
        
def parser():
    parser = argparse.ArgumentParser(description='Make image gallery for image retrieval')
    parser.add_argument('--synset', type=str, required=True, help='Path to class synset txt')
    parser.add_argument('--img', type=str, required=True, help='Path to image root')
    parser.add_argument('--ann', type=str, required=True, help='Path to annotation txt')
    parser.add_argument('--gal', type=str, required=True, help='Path to image gallery')
    parser.add_argument('--n', type=int, default=100, help='Number of adding images per class') 
    args = parser.parse_args()
    return args
def main():
    args = parser()
    class_name = []
    with open(args.synset, "r") as file:
        for line in file:
            class_name.append(line.strip())
    image_gallery_folder_path = args.gal
    if not os.path.exists(image_gallery_folder_path):
        print('Cannot find image gallery folder, now create it')
        os.makedirs(image_gallery_folder_path)
    ann = args.ann
    img_root = args.img
    add_new_data(class_name, image_gallery_folder_path, ann, img_root, args.n)
    

if __name__ == "__main__":
    main()