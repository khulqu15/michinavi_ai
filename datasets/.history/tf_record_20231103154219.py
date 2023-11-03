import tensorflow as tf
import io
import pandas as pd
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple
import os

id_to_class_name = {1: 'Berlubang', 2: 'Retak Buaya', 3: 'Retak Melintang', 4: 'Retak Memanjang'}

def create_tf_example(group, path, label_map):
    group, frame = group 
    class_id = frame['class'].iloc[0]
    if not isinstance(class_id, str):
        class_id = str(class_id)
        
    class_dir = id_to_class_name.get(class_id, 'Unknown')
    if class_dir == 'Unknown':
        print(f"Unknown class id: {class_id}")
        return None
    
    group = str(group)
    locate_data = 'Berlubang'
    full_path = os.path.join('./', class_dir, f'{group}.png')

    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.encode('utf8')  # Using `group` which is the filename.
    image_format = b'png'  # b'jpeg' or b'png'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(label_map[row['class']])

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

def create_tf_record(output_filename, label_map, examples, dataset_directory):
    writer = tf.io.TFRecordWriter(output_filename)
    grouped = examples.groupby('filename')
    for group, frame in grouped:
        group_data = (group, frame)  # Create a tuple of the group and frame.
        tf_example = create_tf_example(group_data, dataset_directory, label_map)
        if tf_example is not None:
            writer.write(tf_example.SerializeToString())
        else:
            print(f"Skipped example for {group} due to unknown class id.")

    writer.close()
    print(f'Successfully created the TFRecord file: {output_filename}')



# Read the label_map.pbtxt file to build the label map dictionary
def load_label_map(label_map_path):
    label_map = {}
    with open(label_map_path, 'r') as file:
        for line in file:
            if "id" in line:
                id = int(line.split(' ')[-1].strip())
            if "name" in line:
                name = line.split(' ')[-1].strip().replace("'", "")
                label_map[name] = id
    return label_map

# Adjust the paths below to match where your files are located
label_map_path = 'label_map.pbtxt'
train_csv_path = 'train_labels.csv'
validation_csv_path = 'validation_labels.csv'
image_dir = os.getcwd()  # The script is assumed to be running from inside the 'datasets' directory

label_map_dict = load_label_map(label_map_path)

# Load the examples from CSV files
train_examples = pd.read_csv(train_csv_path)
validation_examples = pd.read_csv(validation_csv_path)

# Generate the TFRecord files
train_output_path = 'train.record'
validation_output_path = 'validation.record'

create_tf_record(train_output_path, label_map_dict, train_examples, image_dir)
create_tf_record(validation_output_path, label_map_dict, validation_examples, image_dir)
