# Faster R-CNN Captcha Classifier

This is a PyTorch implementation of a Faster R-CNN model for classifying captcha images. The script can be used to train and evaluate the model on a custom COCO-formatted dataset.

## Features
- Loads and preprocesses a COCO-formatted dataset for object detection
- Implements a Faster R-CNN model using PyTorch's `torchvision` library
- Supports training the model with data augmentation (random horizontal flip)
- Saves model checkpoints and the best performing model
- Supports training on GPU (if available)

## Requirements
- Python 3.7 or higher
- PyTorch 1.13.0 or higher
- torchvision 0.14.0 or higher
- numpy
- PIL (Pillow)
- argparse
- json
- os

## Usage

1. Prepare your dataset :
   - Ensure your dataset is in COCO format, with annotations stored in a JSON file and images in a separate directory.
   - Modify the `json_file` and `image_dir` arguments in the command below to point to your dataset.

2. Train the model :
 ``python train_frcnn.py --json_file <path_to_coco_annotations.json> --image_dir <path_to_images_dir> --num_classes <number_of_classes> --output_dir <path_to_output_dir>``
