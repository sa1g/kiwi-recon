"""
Simple and not optimal implementation of a module that automatically loads a 
mask-rcnn model and its parameters.

Params are in `/mask_rcnn/logs/kiwi20240128T1953/mask_rcnn_kiwi_0036.h5`.

When this module is loaded the model is loaded automatically and weights are
applied. The model is exposed through the function `get_masks`. This 
module is been developed thinking that data is sent in batches, not as single
files.
"""

import os
import sys
import cv2
import pickle

ROOT_DIR = os.path.join(os.path.abspath(""), "src", "mask_rcnn")

# Import Mask-RCNN
sys.path.append(ROOT_DIR)
import mrcnn.model as modellib
import kiwi

WEIGHTS_PATH = os.path.join(
    ROOT_DIR, "logs", "kiwi20240128T1953", "mask_rcnn_kiwi_0036.h5"
)

__config = kiwi.KiwiInferenceConfig()
__model = modellib.MaskRCNN(
    mode="inference", config=__config, model_dir=os.path.join(ROOT_DIR, "logs")
)

__model.load_weights(
    WEIGHTS_PATH,
    by_name=True,
)


def segmentation(imgs, bin_idx:int,logs=False):
    """
    Use Mask-RCNN finetuned model to perform instance segmentation on the pictures.

    Args:
        imgs [List(img)]: input list with ONE image to be analyzed.

    Returns:
        List(dict) from Mask-RCNN detection.
    """
    if logs:
        r = __model.detect(imgs)
        with open(f"logs/r-{bin_idx}.pickle", "wb+") as f:
            pickle.dump(r,f)
        return r

    return __model.detect(imgs)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    def __show_image(img, title: str = ""):
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

        plt.axis("off")
        if title != "":
            plt.title(title)
        plt.show()

    def __color_splash(image, mask):
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
        splash = np.where(
            np.sum(mask, axis=-1, keepdims=True) >= 1, splash, gray
        ).astype(np.uint8)

        return splash

    def __detect_and_color_splash(model, image):
        # Detect objects
        r = model.detect([image], verbose=1)[0]

        # Color splash
        splash = __color_splash(image, r["masks"])
        return splash

    IMGS_PATH = os.path.join(os.path.abspath("../"), "dataset", "selected")

    image = cv2.imread(os.path.join(IMGS_PATH, "0.png"))
    # image = skimage.io.imread(os.path.join(IMGS_PATH, "0.png"))
    masked = __detect_and_color_splash(__model, image)

    cv2.imwrite("1.png", image)
    cv2.imwrite("2.png", masked)
