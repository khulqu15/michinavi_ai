"""
A simple script to convert XML files to TFRecord format.
"""

import os
import io
import glob
import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
import xml.etree.ElementTree as ET
from collections import namedtuple, OrderedDict

def xml_to_csv(path):
    """Reads all XML files in the specified directory and converts them to CSV format."""
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

def class_text_to_int(row_label):
    if row_label == 'Berlubang':
        return 1
    elif row_label == 'Retak Buaya':
        return 2
    elif row_label == 'Retak Melintang':
        return 3
    elif row_label == 'Retak Memanjang':
        return 4
    else:
        None

def create_tf_example(group, path):
    """Creates a tf.Example proto from one image and its annotations."""
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
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
        classes.append(class_text_to_int(row['class']))

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
    for directory in ['Berlubang', 'Retak Buaya', 'Retak Melintang', 'Retak Memanjang']:
        image_path = os.path.join(os.getcwd(), './{}'.format(directory))
        xml_df = xml_to_csv(image_path)
        xml_df['class'] = directory  # This line assumes that the directory name is the class name
        grouped = xml_df.groupby('filename')
        writer = tf.io.TFRecordWriter('{}_label.tfrecord'.format(directory))
        for filename, x in grouped:
            tf_example = create_tf_example(x, image_path)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print('Successfully created the TFRecord file for {}.'.format(directory))

if __name__ == '__main__':
    main()
