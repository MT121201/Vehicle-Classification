import fastdup
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Clean images in dataset")
    parser.add_argument("--input", help="Path to the input directory")
    parser.add_argument("--blur", default=200, help="Blur threshold")
    parser.add_argument("--brightness", default=220, help="Brightness threshold")
    parser.add_argument("--darkness", default=50, help="Darkness threshold")
    parser.add_argument("--outliers", default=0.68, help="Outliers threshold")
    args = parser.parse_args()
    return args

def get_clusters(df, sort_by='count', min_count=2, ascending=False):
    # columns to aggregate
    agg_dict = {'filename': list, 'mean_distance': max, 'count': len}

    if 'label' in df.columns:
        agg_dict['label'] = list
    
    # filter by count
    df = df[df['count'] >= min_count]
    
    # group and aggregate columns
    grouped_df = df.groupby('component_id').agg(agg_dict)
    
    # sort
    grouped_df = grouped_df.sort_values(by=[sort_by], ascending=ascending)
    return grouped_df

def main():
    args = parse_args()
    
    fd = fastdup.create(input_dir=args.input)
    fd.run(overwrite=True)
    fd.summary()

    # Get duplicate images
    connected_components_df , _ = fd.connected_components()
    # a function to group connected components
    clusters_df = get_clusters(connected_components_df)
    # First sample from each cluster that is kept
    cluster_images_to_keep = []
    list_of_duplicates = []

    for cluster_file_list in clusters_df.filename:
        # keep first file, discard rest
        keep = cluster_file_list[0]
        discard = cluster_file_list[1:]
        
        cluster_images_to_keep.append(keep)
        list_of_duplicates.extend(discard)

    print(f"Found {len(set(list_of_duplicates))} highly similar images to discard")

    #Get invalid images
    invalid = fd.invalid_instances()
    list_of_invalid_instances = invalid.filename.tolist()
    print(f"Found {len(set(list_of_invalid_instances))} invalid images to discard")

    #Outliers
    outlier_df = fd.outliers()
    theshold_outliers = args.outliers
    list_of_outliers = outlier_df[outlier_df.distance < theshold_outliers].filename_outlier.tolist()
    print(f"Found {len(set(list_of_outliers))} outliers to discard")

    #Blur
    stats_df = fd.img_stats()
    thereshold_blur = args.blur
    blurry_images = stats_df[stats_df['blur'] < thereshold_blur]
    list_of_blurry_images = blurry_images['filename'].to_list()
    print(f"Found {len(set(list_of_blurry_images))} blurry images to discard")

    #Darkness
    thereshold_darkness = args.darkness
    dark_images = stats_df[stats_df['mean'] < thereshold_darkness]  
    list_of_dark_images = dark_images['filename'].to_list()
    print(f"Found {len(set(list_of_dark_images))} dark images to discard")

    #Brightness
    thereshold_brightness = args.brightness
    bright_images = stats_df[stats_df['mean'] > thereshold_brightness]
    list_of_bright_images = bright_images['filename'].to_list()
    print(f"Found {len(set(list_of_bright_images))} bright images to discard")

    image_paths = list_of_dark_images + list_of_bright_images + list_of_blurry_images + list_of_outliers + list_of_invalid_instances + list_of_duplicates
    print(f"Total files to delete: {len(image_paths)}")

    deleted_count = 0

    for file_path in image_paths:
        try:
            os.remove(file_path)
            deleted_count += 1
        except:
            pass

    print(f"Total files deleted: {deleted_count}")

if __name__ == "__main__":  
    main()