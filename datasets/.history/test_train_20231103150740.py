import os
import io
from PIL import Image
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
from sklearn.model_selection import train_test_split
from object_detection.utils import dataset_util
from collections import namedtuple

# Adjust these paths as necessary
data_dir = 'datasets/'
output_dir = 'output_tfrecords/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the class mapping
class_dict = {
    'Berlubang': 1,
    'Retak Buaya': 2,
    'Retak Melintang': 3,
    'Retak Memanjang': 4
}

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def split_df(df, test_size=0.2):
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df, val_df

def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_png)
    image = Image.open(encoded_png_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'png'
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
        classes.append(class_dict[row['class']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_png),
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
    for class_dir in ['Berlubang', 'Retak Buaya', 'Retak Melintang', 'Retak Memanjang']:
        class_path = os.path.join(data_dir, class_dir)
        xml_df = xml_to_csv(class_path)
        train_df, val_df = split_df(xml_df)
        train_output_path = os.path.join(output_dir, class_dir + '_train.record')
        val_output_path = os.path.join(output_dir, class_dir + '_val.record')
        
        # Create train record
        writer = tf.io.TFRecordWriter(train_output_path)
        grouped = train_df.groupby('filename')
        for filename, group in grouped:
            tf_example = create_tf_example(group, class_path)
            writer.write(tf_example.SerializeToString())
        writer.close()

        # Create val record
        writer = tf.io.TFRecordWriter(val_output_path)
        grouped = val_df.groupby('filename')
        for filename, group in grouped:
            tf_example = create_tf_example(group, class_path)
            writer.write(tf_example.SerializeToString())
        writer.close()

if __name__ == '__main__':
    main()
