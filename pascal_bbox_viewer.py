"""
This script is for viewing bounding box annotations in Pascal VOC format
The script assumes xmls and images are placed in the same directory
"""

import cv2
import bbox_operations
import argparse
import os


def draw_annotations(img, labels, bboxes):
    '''
    Function that draws the bounding box annotations
    with labels
    '''

    for i, bbox in enumerate(bboxes):
        
        # pt1 = upper left corner, pt2 = lower right corner
        img = cv2.rectangle(img, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int( bbox[3])), (255,0,0), 2)

        # Draw the label
        img = cv2.putText(img,labels[i],(int(bbox[0]),int(bbox[1]-5)),0,0.5,(255,0,0),2,cv2.LINE_AA)
    
    return img


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('base_path',help='the base path that contains images and xmls')

    args = argparser.parse_args()

    for img_file in os.listdir(args.base_path):

        if img_file.endswith(("jpg", 'jpeg', 'png', 'JPEG', 'JPG')):

            filename = img_file.split('.')[0]
            
            xml_filename = filename+'.xml'
            xml_filepath = os.path.join(args.base_path, xml_filename)
            
            image_filepath = os.path.join(args.base_path, img_file)
            
            
            with open(xml_filepath, 'r') as f:
                boxes, labels = bbox_operations.parse_pascalvoc_bboxes_from_xml(f)
            
            img = cv2.imread(image_filepath)
            img = draw_annotations(img, labels, boxes)
               
            cv2.imshow('image', img)
            if cv2.waitKey(0) % 256 == ord('q'):
                break
