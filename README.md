# BBOX_tools 

This repository is for performing data augmentation operations to bounding box annotations in Pascal VOC format.

## Requirements

* Python 3.7
* [Albumentations Library](https://github.com/albu/albumentations)

### Augmentation script
In order to perform predefined augmentation stack to pascal voc formatted images, run the script:
```
python3 augmentor.py <path_to_files> --no_augmentations
```
The default number of augmentations for each original sample is 5. The generated files are saved under the input folder.