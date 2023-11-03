import tensorflow as tf
import pandas as pd
import io
import os
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

# Convert the label map to a dictionary
def load_label_map(label_map_path):
    label_map_dict = {}
    with open(label_map_path, 'r') as file:
        for line in file:
            if "id" in line:
                id = int(line.split(":")[1])
                name = next(file).split("'")[1]
                label_map_dict[name] = id
    return label_map_dict

# Create a TensorFlow Example from a single data point
def create_tf_example(group, path, label_map_dict):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'png'
    xmins = [group.xmin / width]
    xmaxs = [group.xmax / width]
    ymins = [group.ymin / height]
    ymaxs = [group.ymax / height]
    classes_text = [group['class'].encode('utf8')]
    classes = [label_map_dict[group['class']]]

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

# Main function to generate the TFRecord file
def generate_tfrecord(csv_input, image_dir, output_path, label_map_dict):
    writer = tf.io.TFRecordWriter(output_path)
    examples = pd.read_csv(csv_input)
    grouped = examples.groupby('filename')
    for filename, group in grouped:
        tf_example = create_tf_example(group, image_dir, label_map_dict)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print(f'Successfully created the TFRecord file: {output_path}')

def main():
    # Path to the CSV input
    train_csv_input = 'train_labels.csv'
    validation_csv_input = 'validation_labels.csv'
    # Path to the image directory
    train_image_dir = 'datasets/'
    validation_image_dir = 'datasets/'
    # Path to output TFRecord files
    train_output_path = 'train.record'
    validation_output_path = 'validation.record'
    # Path to label map file
    label_map_path = 'label_map.pbtxt'

    # Load label map
    label_map_dict = load_label_map(label_map_path)

    # Generate TFRecord for the training dataset
    generate_tfrecord(train_csv_input, train_image_dir, train_output_path, label_map_dict)
    # Generate TFRecord for the validation dataset
    generate_tfrecord(validation_csv_input, validation_image_dir, validation_output_path, label_map_dict)

if __name__ == '__main__':
    main()
