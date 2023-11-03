import xml.etree.ElementTree as ET
import csv
import os
import glob

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
    return xml_list, column_name

def main():
    for directory in ['Berlubang', 'Retak Buaya', 'Retak Melintang', 'Retak Memanjang']:
        image_path = os.path.join(os.getcwd(), ('datasets/' + directory))
        xml_list, column_name = xml_to_csv(image_path)
        csv_filename = './train_labels.csv'
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(column_name)
            for row in xml_list:
                writer.writerow(row)
        print('Successfully converted xml to csv.')

main()
