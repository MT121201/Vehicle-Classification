##############################################################################################################
# TODO: Is this description correct?? This script randomly splits a txt file into train and test files. The input txt file should have the format:
#     image_path class_index
#     image_path class_index
# Usage:
#     python tools/check_data/devide_dataset.py \
#         --input orginal annotations file \
#         --thresh split threshold, default is 0.85 \
#         --out output directory, default is in the same directory with input file
##############################################################################################################
import os
import random
import argparse

def remove_empty_lines(file_path):
    # Removes empty lines from a file and returns the non-empty lines as a list
    with open(file_path, 'r') as f:
        lines = f.readlines()
    non_empty_lines = [line for line in lines if line.strip()]
    return non_empty_lines

def split_txt_file(input_file, threshold, train_file, test_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Filter out empty lines and lines that don't have class index
    valid_lines = [line for line in lines if line.strip() and line.strip().split()[-1].isdigit()]
    
    random.shuffle(valid_lines)
    total_lines = len(valid_lines)
    train_lines = valid_lines[:int(threshold * total_lines)]
    test_lines = valid_lines[int(threshold * total_lines):]
    
    with open(train_file, 'w') as train_f:
        train_f.writelines(train_lines)
        
    with open(test_file, 'w') as test_f:
        test_f.writelines(test_lines)

def sort_by_class(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    sorted_lines = sorted(lines, key=lambda line: int(line.strip().split()[-1]))
    with open(file_path, 'w') as f:
        f.writelines(sorted_lines)

def arg_parse():
    parser = argparse.ArgumentParser(description="Randomly split a txt file into train and test files.")
    parser.add_argument("--input", help="Path to the input txt file")
    parser.add_argument("--thresh", type=float, default=0.85, help="Threshold for the random split (0.0 to 1.0)")
    parser.add_argument("--out",default=None ,help="Path to the output directory")
    args = parser.parse_args()
    return args
def main():
    args = arg_parse()
    # Check if user has provided the output directory, if not use the same directory as the input file
    if args.out is None:
        output_directory = os.path.dirname(args.input_file)
    else:
        output_directory = args.out
    # Check if the output directory exists, if not create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Create the train.txt and test.txt file paths
    train_txt_file = os.path.join(output_directory, "train.txt")
    test_txt_file = os.path.join(output_directory, "test.txt")

    # Perform the random split and save to train.txt and test.txt
    split_txt_file(args.input, args.thresh, train_txt_file, test_txt_file)

    # Remove empty lines from the output files
    non_empty_train_lines = remove_empty_lines(train_txt_file)
    with open(train_txt_file, 'w') as train_f:
        train_f.writelines(non_empty_train_lines)

    non_empty_test_lines = remove_empty_lines(test_txt_file)
    with open(test_txt_file, 'w') as test_f:
        test_f.writelines(non_empty_test_lines)

    sort_by_class(train_txt_file)
    sort_by_class(test_txt_file)
    print("Random split completed successfully!")

if __name__ == '__main__':
    main()
