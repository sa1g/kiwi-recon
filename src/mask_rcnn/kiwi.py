"""
Mask-RCNN
Train on the kiwi segmentation dataset

Written by Ettore Saggiorato, inspired by `nucleus.py`

----

Usage: TODO
"""

import datetime
import json
import os
import sys
import numpy as np
from imgaug import augmenters as iaa
import cv2

ROOT_DIR = os.path.abspath("../../")
import skimage.draw

# Import Mask-RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize


# Path to pre-trained Mask-RCNN weights file (COCO)
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Default directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class KiwiConfig(Config):
    """
    Configuration for training on the kiwis segmentation dataset.
    """

    # Give the config a recognizable name
    NAME = "kiwi"

    # We are using a 24GB VRAM GPU, 4 images should fit in the memory
    IMAGES_PER_GPU = 4

    NUM_CLASSES = 1 + 1  # Background + kiwi

    # TODO: manual injection
    STEPS_PER_EPOCH = 250
    VALIDATION_STEPS = 25

    # Don't exclude based on confidence. Since we have two classes then 0.5 is the minimum anyway as it picks between kiwi and BG
    DETECTION_MIN_CONFIDENCE = 0

    BACKBONE = "resnet101"

    # Non-max suppression threshold to filter RPN proposals.You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # If enabled, resizes instance masks to a smaller size to reduce memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)

    # Input image resizing
    # Random crops of size 512x512 (as it seems fair enough) - useful for big imgs
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per iamge
    DETECTION_MAX_INSTANCES = 400


class KiwiInferenceConfig(KiwiConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################


class KiwiDataset(utils.Dataset):
    """
    Adapted from `baloon.py`
    """

    def load_kiwi(self, dataset_dir, subset):
        """
        Load a subset of the kiwi dataset.

        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("kiwi", 1, "kiwi")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(
            open(os.path.join(dataset_dir, "via_export_json.json"), encoding="utf-8")
        )
        annotations = list(annotations.values())

        # The VIA tool saves images in the JSON even if they don't have any annotations. Skip unannotated images.
        annotations = [a for a in annotations if a["regions"]]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up the outline of each object instance. These are stores in the shape_attributes. The if condition is needed to support VIA versions 1.x and 2.x.
            if isinstance(a["regions"], dict):
                polygons = [r["shape_attributes"] for r in a["regions"].values()]
            else:
                polygons = [r["shape_attributes"] for r in a["regions"]]

            # load_mask() needs the image size to convert polygons to masks. Unfortunately, VIA doesn't include it in JSON, so we must read the image. This is only manageable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a["filename"])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "kiwi",
                image_id=a["filename"],  # use file name as a unique image id
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.

        Returns:
            masks: A bool array of shape [height, width, instance count] with one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a kiwi dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "kiwi":
            return super().load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros(
            [info["height"], info["width"], len(info["polygons"])], dtype=np.uint8
        )

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p["all_points_y"], p["all_points_x"])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have one class ID only, we return an array of 1s
        return mask.astype(np.bool__), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "kiwi":
            return info["path"]
        else:
            super().image_reference(image_id)


############################################################
#  Training
############################################################


def train(model, dataset_dir, subset):
    """
    Train the model
    """
    # Training dataset
    dataset_train = KiwiDataset()
    dataset_train.load_kiwi(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = KiwiDataset()
    dataset_val.load_kiwi(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    augmentation = iaa.SomeOf(
        (0, 2),
        [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf(
                [iaa.Affine(rotate=90), iaa.Affine(rotate=180), iaa.Affine(rotate=270)]
            ),
            iaa.Multiply((0.8, 1.5)),
            iaa.GaussianBlur(sigma=(0.0, 5.0)),
        ],
    )

    # TODO: fix to my needs
    print("Training network heads")
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=50,
        augmentation=augmentation,
        layers="heads",
    )

    # IDK why after loading the models it doesn't train
    # print("Training all layers")
    # model.train(
    #     dataset_train,
    #     dataset_val,
    #     learning_rate=config.LEARNING_RATE,
    #     epochs=40,
    #     augmentation=augmentation,
    #     layers="all",
    # )


############################################################
#  Color Splash
############################################################
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    splash = np.zeros_like(image)

    for i in range(mask.shape[-1]):
        color = np.random.randint(0, 256, 3)  # Random RGB color
        mask_single = mask[:, :, i]

        # Convert boolean mask to uint8
        mask_single = mask_single.astype(np.uint8)

        # Create a color mask
        color_mask = np.zeros_like(image)
        color_mask[:, :, :] = color

        # Apply color mask to the splash image
        splash += cv2.bitwise_and(
            color_mask, color_mask, mask=mask_single[:, :, np.newaxis]
        )

    # Combine the colored mask with the grayscale image
    splash = np.where(np.sum(mask, axis=-1, keepdims=True) >= 1, splash, gray).astype(
        np.uint8
    )

    return splash


def detect_and_color_splash(model, image_path=None):
    assert image_path

    # Run model detection and generate the color splash effect
    print("Running on {}".format(args.image))

    # Read image
    image = skimage.io.imread(args.image)

    # Detect objects
    r = model.detect([image], verbose=1)[0]

    # Color splash
    splash = color_splash(image, r["masks"])

    # Save output
    file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    skimage.io.imsave(file_name, splash)

    print("Saved to ", file_name)


############################################################
#  Command Line
############################################################

if __name__ == "__main__":
    import argparse

    # Parse command line args
    parser = argparse.ArgumentParser(description="Train Mask-RCNN to detect kiwis.")
    parser.add_argument("command", metavar="<command>", help="'train' or 'splash'")
    parser.add_argument(
        "--dataset",
        required=False,
        metavar="/path/to/kiwi/dataset/",
        help="Directory of the kiwi dataset",
    )
    parser.add_argument(
        "--weights",
        required=True,
        metavar="/path/to/weights.h5",
        help="Path to weights .h5 file or 'coco'",
    )
    parser.add_argument(
        "--logs",
        required=False,
        default=DEFAULT_LOGS_DIR,
        metavar="/path/to/logs/",
        help="Logs and checkpoints directory (default=logs/)",
    )
    parser.add_argument(
        "--image",
        required=False,
        metavar="path or URL to image",
        help="Image to apply the color splash effect on",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image, "Provide --image to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = KiwiConfig()
    else:
        config = KiwiInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(
            weights_path,
            by_name=True,
            exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"],
        )
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, "train")
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image)
    else:
        print("'{}' is not recognized. " "Use 'train' or 'splash'".format(args.command))
