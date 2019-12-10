"""
This script performs image augmentations for object detection using albumentations library.
"""

from urllib.request import urlopen
import os

import numpy as np
import cv2
from matplotlib import pyplot as plt

import albumentations as A

from bbox_operations import parse_pascalvoc_bboxes_from_xml, write_pascalvoc_xml
import argparse
import os
import pickle

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def visualize_bbox_pascal_voc(img, bbox, class_id, class_idx_to_name="", color=BOX_COLOR, thickness=2):
    """Visualize bounding boxes in Pascal VOC format"""
    x_min, y_min, x_max, y_max = bbox
    # x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
    class_name = class_id   #class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    #cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, class_name, (int(x_min), int(y_min) - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR, lineType=cv2.LINE_AA)
    return img


def visualize(annotations, category_id_to_name=""):
    """Visualize bboxes on image """
    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
        img = visualize_bbox_pascal_voc(img, bbox, annotations['category_id'][idx])
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.show()


def get_aug(aug, min_area=0., min_visibility=0.):
    return A.Compose(aug, bbox_params={'format': 'pascal_voc', 'min_area': min_area, 'min_visibility': min_visibility, 'label_fields': ['category_id']})


def check_output_folder(path):

    if(os.path.exists(path)):
        print('Path exists')
        return True
    else:
        os.mkdir(path)
        return True


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("source_path", help="Path of the source image and annotation files")
    argparser.add_argument("--no_augmentations", type=int, default=5, 
                           help="No of augmented examples for each original sample")
    argparser.add_argument('--save_augmentor', type=bool, default=False)
    argparser.add_argument('--augmentor_file', type=str, default=None)
    argparser.add_argument('--destination_path', type=str, default=None)
    
    args = argparser.parse_args()
    img_files = []
    xml_files = []

    img_filenames = []

    assert os.path.exists(args.source_path) is True

    if args.destination_path is None:
        augmentation_path = os.path.join(args.source_path, "augmentations")
    else:
        augmentation_path = args.destination_path

    check_output_folder(augmentation_path)

    for file in os.listdir(args.source_path):
        if file.endswith((".jpg", 'png', 'jpeg')):
            img_filenames.append(file)
            img_path = os.path.join(args.source_path, file)
            xml_path = img_path[:-3]
            xml_path += "xml"
            img_files.append(img_path)
            xml_files.append(xml_path)

    for i, afile in enumerate(img_files):
           
        for j in range(0, args.no_augmentations):
            img_file = afile.split("/")[-1]
            xml_file = xml_files[i].split("/")[-1]

            new_file = img_file[:-4]
            new_file += "_"
            new_file += str(j)

            new_img_file = new_file+img_file[-4:]
            new_img_filename = new_img_file.split("/")[-1]
                
            new_xml_file = new_file+xml_file[-4:]
            new_xml_filename = new_xml_file.split("/")[-1]

            new_img_file = os.path.join(augmentation_path, new_img_filename)
            new_xml_file = os.path.join(augmentation_path, new_xml_filename)

            print(new_img_file)
            print(new_xml_file)

            # continue

            bboxes, cat_ids = parse_pascalvoc_bboxes_from_xml(xml_files[i])
            image = cv2.imread(afile,cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            height = image.shape[0]
            width = image.shape[1]
            wh = width/height

            annotations = {'image': image.copy(), 'bboxes': bboxes, 'category_id': cat_ids}

            aug = None

            if args.augmentor_file is not None:
                f = open(args.augmentor_file, 'rb')
                aug = pickle.load(f)
                f.close()
            else:
                aug = get_aug([
                            A.RandomSizedCrop((int(height/4),int(height*0.5)),int(height/2),int(wh*(height/2)),p=0.5),
                            A.ShiftScaleRotate(shift_limit=0,scale_limit=0.,rotate_limit=3,p=0.5),
                            #A.RGBShift(p=0.5),
                            A.Blur(blur_limit=2, p=0.5),
                            A.RandomBrightness(p=0.5),
                            #A.CLAHE(p=0.5),
                            ])
            
            if args.save_augmentor:
                f = open('augmentor.pickle', 'wb')
                pickle.dump(aug, f)
                f.close()
            
            augmented = aug(**annotations)
            
            #visualize(augmented)
            
            img = augmented["image"].copy()
            
            height, width, channels = img.shape
            
            write_pascalvoc_xml(augmented["bboxes"], augmented["category_id"], height, width,
                                new_xml_file, new_img_filename)
            
            img = augmented["image"].copy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(new_img_file, img)

        #print(cat_ids)
        #print(bboxes)
