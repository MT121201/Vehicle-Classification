import argparse
import os
import cv2
import matplotlib.pyplot as plt
import random
import math

def visualize_images(annotation_file, image_root, class_index, num_images, output_file, show):
    with open(annotation_file, 'r') as f:
        annotations = f.readlines()

    class_images = [line.strip().split() for line in annotations if line.strip().endswith(str(class_index))]

    if not class_images:
        print(f"No images found for class index {class_index}.")
        return

    num_rows = math.isqrt(num_images)
    num_cols = math.ceil(num_images / num_rows)

    plt.figure(figsize=(15, 15))

    for i in range(num_images):
        random_image_info = random.choice(class_images)
        image_name = random_image_info[0]
        image_path = os.path.join(image_root, image_name)
        if not os.path.exists(image_path):
            print(f"Image {image_path} does not exist.")
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(image)
            plt.title(f"Image {i+1}")
            plt.axis('off')

    plt.tight_layout()
    # save the figure
    if show:
        print("Detected --show flag, showing the image")
        plt.show()
    
    plt.savefig(output_file)
    print(f"Saved visualization to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualize random images from a specific class.')
    parser.add_argument('annotation_file', type=str, help='Path to the annotation file')
    parser.add_argument('image_root', type=str, help='Root directory of the images')
    parser.add_argument('--c', type=int, help='Class index to visualize')
    parser.add_argument("--o",default=None ,help="Output image")
    parser.add_argument('--show', action='store_true', help='Show the plotted images')
    parser.add_argument('--n',default=50 ,type=int, help='Number of images to visualize')
    args = parser.parse_args()

    if not os.path.exists(args.annotation_file):
        print(f"Annotation file {args.annotation_file} does not exist.")
        return
    if not os.path.exists(args.image_root):
        print(f"Image root {args.image_root} does not exist.")
        return

    if args.c is None:
        print("Class index not provided.")
        return
    
    if args.o is None:
        print('No output image provided, will save to default name')
        output_file = "./cache/" + 'Visualize_class_' + str(args.c) + '.jpg'
    else:
        output_file = args.o
    visualize_images(args.annotation_file, args.image_root, args.c, args.n, output_file, args.show)

if __name__ == '__main__':
    main()