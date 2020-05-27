# BBOX_tools 

This repository is for performing data augmentation operations to bounding box annotations in Pascal VOC format.

## Requirements

* Python 3.7
* [Albumentations Library](https://github.com/albu/albumentations)
* OpenCV >= 3

### Augmentation script
In order to perform predefined augmentation stack to pascal voc formatted images, run the script:
```
python3 augmentor.py <path_to_files> --no_augmentations
```
The default number of augmentations for each original sample is 5. The generated files are saved under the input folder.

### Pascal BBox Viewer
The bounding box annotations in Pascal format can be displayed using the display script. Run the following command to display the annotations
```
python3 pascal_bbox_viewer.py <base_path>
```
where `<base_path>` contains the xmls and images. If you want to quit the visualization loop, press `q`.

