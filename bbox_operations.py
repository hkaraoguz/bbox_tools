"""
This script contains operations for handling bounding box data in pascal VOC format.
"""

import xml.etree.ElementTree as ET
import argparse
import os
from xml.dom import minidom


def write_csv_bbox(bboxes, category_ids, csv_filename, image_path):
    """Creates a csv file from Pascal VOC bounding boxes.
    Especially for https://github.com/hkaraoguz/pytorch-retinanet

    Args:
      bboxes: array of bounding boxes. Bboxes are in xmin, ymin, xmax, ymax
           format
      category_ids: the category ids of the boundingboxes
      csv_filename: filename of the output csv
      image_path: relative path of the image
    """
    if len(bboxes) == 0:
        with open(csv_filename, 'a') as f:
            f.write("{},,,,,\n".format(image_path))
        return

    with open(csv_filename, 'a') as f:
        for i, bbox in enumerate(bboxes):

            x_min, y_min, x_max, y_max = bbox
            x_min = int(x_min)
            y_min = int(y_min)
            x_max = int(x_max)
            y_max = int(y_max)

            f.write("{},{},{},{},{},{}\n".format(image_path, x_min, y_min, x_max, y_max, category_ids[i]))


def write_pascalvoc_xml(bboxes, category_ids, xml_filename, img_filename):
    """Writes bboxes in Pascal VOC format.

    Args:
      bboxes: array of bounding boxes. Bboxes are in xmin, ymin, xmax, ymax
           format
      category_ids: the category ids of the boundingboxes
      xml_filename: filename of the output xml
      image_filename: name of the image file
    """
    root = ET.Element("annotation")
    x = ET.SubElement(root, "filename")
    x.text = img_filename

    # For each bbox, create xml entry
    for i, bbox in enumerate(bboxes):
        obj = ET.SubElement(root, "object")
        name = ET.SubElement(obj, "name")
        name.text = category_ids[i]
        x = ET.SubElement(obj, "pose")
        x.text = "Unspecified"
        x = ET.SubElement(obj, "truncated")
        x.text = "0"
        x = ET.SubElement(obj, "difficult")
        x.text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        x_min, y_min, x_max, y_max = bbox
        
        # Convert values to int
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)

        x = ET.SubElement(bndbox, "xmin")
        x.text = str(x_min)
        x = ET.SubElement(bndbox, "ymin")
        x.text = str(y_min)
        x = ET.SubElement(bndbox, "xmax")
        x.text = str(x_max)
        x = ET.SubElement(bndbox, "ymax")
        x.text = str(y_max)
        
    tree = ET.ElementTree(root)

    # Output the xml in pretty format
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")

    with open(xml_filename, "w") as f:
        f.write(xmlstr)


def parse_pascalvoc_bboxes_from_xml(file):
    """Parses bboxes in Pascal VOC format from XML files.

    Args:
      file: XML file to be parsed
    Returns:
      bboxes: bounding boxes array
      category_ids: category array of bboxes
    """
    bboxes = []
    category_ids = []
    
    try:
        tree = ET.parse(file)
    except Exception as ex:
        print(ex)
        return bboxes, category_ids

    root = tree.getroot()
    for obj in root.iter('object'):
        bbox = []
        for bndbox in obj.iter("bndbox"):
            bbox.append(float(bndbox[0].text))
            bbox.append(float(bndbox[1].text))
            bbox.append(float(bndbox[2].text))
            bbox.append(float(bndbox[3].text))
            bboxes.append(bbox)
        category_ids.append(str(obj[0].text))

    return bboxes, category_ids


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()

    argparser.add_argument("path")

    args = argparser.parse_args()

    files = []

    csv_path = os.path.join(args.path, "training.csv")

    for file in os.listdir(args.path):
        if file.endswith(".xml"):
            print(os.path.join(args.path, file))
            files.append(os.path.join(args.path, file))

    for afile in files:
        bboxes, cat_ids = parse_pascalvoc_bboxes_from_xml(afile)
        img_path = afile[:-3]
        img_path += "jpg"
        print(img_path)
        print(csv_path)
        write_csv_bbox(bboxes, cat_ids, csv_path, img_path)
