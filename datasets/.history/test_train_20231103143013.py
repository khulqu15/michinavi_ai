import os
import glob
import random
import shutil
from xml_to_csv import xml_to_csv  # Assuming you've saved the previous script with this function

def split_data_set(image_dir, train_ratio=0.8):
    # Get the list of all image files and their corresponding XML files
    images = glob.glob(os.path.join(image_dir, '*.png'))
    xmls = glob.glob(os.path.join(image_dir, '*.xml'))
    
    # Pair them up by sorting
    images.sort()
    xmls.sort()

    # Compute split sizes
    total_images = len(images)
    train_size = int(total_images * train_ratio)

    # Shuffle the images and xmls in the same order
    paired_images_xmls = list(zip(images, xmls))
    random.shuffle(paired_images_xmls)
    train_pairs = paired_images_xmls[:train_size]
    val_pairs = paired_images_xmls[train_size:]

    # Move files
    for pair in train_pairs:
        shutil.move(pair[0], 'train/')
        shutil.move(pair[1], 'train/')

    for pair in val_pairs:
        shutil.move(pair[0], 'val/')
        shutil.move(pair[1], 'val/')

# Run the split for each class directory
dataset_dir = 'datasets'
classes = ['Berlubang', 'Retak Buaya', 'Retak Melintang', 'Retak Memanjang']

for class_dir in classes:
    image_dir = os.path.join(dataset_dir, class_dir)
    split_data_set(image_dir)

    # Convert annotations to CSV
    train_csv = xml_to_csv(os.path.join('train', class_dir))
    val_csv = xml_to_csv(os.path.join('val', class_dir))
    
    # Save CSV files
    train_csv.to_csv(f'{class_dir}_train_labels.csv', index=False)
    val_csv.to_csv(f'{class_dir}_val_labels.csv', index=False)
