import os

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip().split() for line in lines]

def calculate_statistics(data):
    total_images = len(data)
    class_counts = {}
    for _, class_idx in data:
        class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

    class_percentages = {class_idx: (count / total_images) * 100 for class_idx, count in class_counts.items()}
    return total_images, class_counts, class_percentages

def calculate_balancing_factor(total_images, num_classes, class_counts):
    target_images_per_class = total_images / num_classes
    balancing_factors = {}
    for class_idx, count in class_counts.items():
        factor = target_images_per_class / count
        balancing_factors[class_idx] = factor
    return balancing_factors

def main(file_paths, num_classes):
    data = []
    for file_path in file_paths:
        data.extend(read_txt_file(file_path))

    total_images, class_counts, class_percentages = calculate_statistics(data)

    print("Total Images:", total_images)
    print("Number of Images per Class:")
    for class_idx, count in class_counts.items():
        print(f"Class {class_idx}: {count} images")
    
    print("Percentage of Each Class:")
    for class_idx, percentage in class_percentages.items():
        print(f"Class {class_idx}: {percentage:.2f}%")

    balancing_factors = calculate_balancing_factor(total_images, num_classes, class_counts)

    print("\nBalancing Factors:")
    for class_idx, factor in balancing_factors.items():
        print(f"Class {class_idx}: {factor} ")

if __name__ == '__main__':
    file_paths = ['/data/its/vehicle_cls/202307_crop_ttp/annotations/train.txt', '/data/its/vehicle_cls/vp3_202307_crop/annotations/train.txt']  # Replace with the paths of your text files
    num_classes = 10  # Replace with the actual number of classes in your dataset
    main(file_paths, num_classes)