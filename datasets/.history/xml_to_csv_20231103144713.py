import xml.etree.ElementTree as ET
import pandas as pd
import os

dataset_dir = './'

csv_file = './train_labels.csv'

class_name_to_id = {
    'Berlubang': 1,
    'Retak Buaya': 2,
    'Retak Melintang': 3,
    'Retak Memanjang': 4
}

# Function to parse a single XML file
def xml_to_csv(xml_file, folder):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    xml_list = []
    
    for member in root.findall('object'):
        value = (root.find('filename').text,
                 int(root.find('size')[0].text),
                 int(root.find('size')[1].text),
                 class_name_to_id[member[0].text],
                 int(member[4][0].text),
                 int(member[4][1].text),
                 int(member[4][2].text),
                 int(member[4][3].text)
                 )
        xml_list.append(value)
    return xml_list

# Function to iterate over all folders and XML files
def process_dataset(dataset_dir, csv_file):
    xml_list = []
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.isdir(folder_path):
            for xml_file in os.listdir(folder_path):
                if xml_file.endswith('.xml'):
                    xml_path = os.path.join(folder_path, xml_file)
                    xml_list.extend(xml_to_csv(xml_path, folder))

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df.to_csv(csv_file, index=None)
    print(f'Successfully converted XML to CSV and saved to {csv_file}')

# Run the conversion process
process_dataset(dataset_dir, csv_file)
