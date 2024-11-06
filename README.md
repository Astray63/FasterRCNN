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
- `--json_file`: Path to the COCO format annotation file.
- `--image_dir`: Directory containing the images.
- `--num_classes`: Number of classes including the background class.
- `--output_dir`: Directory to save model checkpoints.
- `--num_epochs`: Number of training epochs (default is 50).
- `--batch_size`: Batch size for training (default is 2).

3. Monitor the training:
- The script will output the average loss for each epoch.
- The best performing model will be saved to `<output_dir>/frcnn_captcha_best.pth`.
- Checkpoints will be saved every 10 epochs to `<output_dir>/frcnn_captcha_epoch_<epoch>.pth`.
- If the training is interrupted, the current model will be saved to `<output_dir>/frcnn_captcha_interrupted.pth`.
- The final model will be saved to `<output_dir>/frcnn_captcha_final.pth`.

## Customization
- You can modify the data augmentation techniques by updating the `get_transform` function.
- The Faster R-CNN model architecture can be customized by modifying the `fasterrcnn_resnet50_fpn_v2` function call.
- The optimizer, learning rate scheduler, and training hyperparameters can be adjusted as needed.
