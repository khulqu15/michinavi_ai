import tensorflow as tf
import pandas as pd
import argparse
import logging
import io
import os
import xml.etree.ElementTree as ET

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def create_example(xml_file, path):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Start building the TFRecord Example
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    with tf.io.gfile.GFile(os.path.join(path, filename), 'rb') as fid:
        encoded_image = fid.read()

    image_format = b'png' # or b'jpeg' if the images are in jpeg format
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    # Read the labelmap
    class_names = read_labelmap('label_map.pbtxt')

    for member in root.findall('object'):
        classes_text.append(member[0].text.encode('utf8'))
        classes.append(class_names[member[0].text])
        
        bndbox = member.find('bndbox')
        xmins.append(float(bndbox.find('xmin').text) / width)
        xmaxs.append(float(bndbox.find('xmax').text) / width)
        ymins.append(float(bndbox.find('ymin').text) / height)
        ymaxs.append(float(bndbox.find('ymax').text) / height)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return example

def read_labelmap(labelmap_path):
    with open(labelmap_path, 'r') as file:
        labelmap = file.read()
    
    items = labelmap.split('item')[1:]
    class_names = {}
    for item in items:
        name = str(item.split('name')[1].split('"')[1])
        name_id = int(item.split('name')[1].split('id')[1].split(": ")[1].split('}')[0])
        class_names[name] = name_id
    return class_names

def main(dataset_path, labelmap_path):
    class_names = read_labelmap(labelmap_path)
    for class_dir in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, class_dir)):
            class_path = os.path.join(dataset_path, class_dir)
            output_file = os.path.join(dataset_path, class_dir + '.record')
            writer = tf.io.TFRecordWriter(output_file)
            for xml_file in os.listdir(class_path):
                if xml_file.endswith('.xml'):
                    xml_path = os.path.join(class_path, xml_file)
                    tf_example = create_example(xml_path, class_path)
                    writer.write(tf_example.SerializeToString())
            writer.close()
            logging.info(f"Successfully created the TFRecord file: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tf record for each class directory.")
    parser.add_argument('--dataset_path', help='Path to the dataset directory', default='datasets/')
    parser.add_argument('--labelmap_path', help='Path to the labelmap.pbtxt file', default='label_map.pbtxt')
    args = parser.parse_args()
    
    main(args.dataset_path, args.labelmap_path)
