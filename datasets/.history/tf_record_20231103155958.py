import os
import pandas as pd
import tensorflow as tf
from object_detection.utils import dataset_util
from PIL import Image
from collections import namedtuple

# Adjust this function according to how your image data is organized
def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, dataset_path):
    full_path = os.path.join(dataset_path, group.filename)
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_image_io = tf.io.gfile.GFile(full_path, 'rb')
    image = Image.open(encoded_image_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'png'  # Or b'jpeg' or b'jpg' for JPEG images

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
               # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
               # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(row['class_id'])  # Changed to 'class_id'

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main():
    # Adjust these paths to your dataset's annotation files and image folder
    train_csv_input = './train_labels.csv'
    validation_csv_input = './validation_labels.csv'
    train_output_path = './train.record'
    validation_output_path = './validation.record'
    dataset_path = './'  # Base directory for the dataset

    # Create TFRecord for training images
    train_examples = pd.read_csv(train_csv_input)
    grouped_train = split(train_examples, 'filename')
    with tf.io.TFRecordWriter(train_output_path) as writer:
        for group in grouped_train:
            tf_example = create_tf_example(group, dataset_path)
            writer.write(tf_example.SerializeToString())

    # Create TFRecord for validation images
    validation_examples = pd.read_csv(validation_csv_input)
    grouped_validation = split(validation_examples, 'filename')
    with tf.io.TFRecordWriter(validation_output_path) as writer:
        for group in grouped_validation:
            tf_example = create_tf_example(group, dataset_path)
            writer.write(tf_example.SerializeToString())

    print('Successfully created the TFRecords: {} and {}'.format(train_output_path, validation_output_path))

if __name__ == '__main__':
    main()
