"""
Run this from its directory.

For info read README.md
"""

import os
import cv2
import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import logging

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)

from src.camera_calibration import get_parameters
from src.bin_detection import bin_crop
from src.instance_segmentation import segmentation
from src.get_size import get_ellipses_sizes
from src.size_class import converter


if __name__ == "__main__":
    # Get file names
    file_names = glob.glob(
        os.path.join(os.path.abspath(""), "datasets", "raw_selected", "*.jpg")
    )
    # Get undistort parameters
    mtx, dist, newcameramtx, roi = get_parameters()

    classes = np.array([0] * 11)
    classes1 = np.array([0] * 11)
    for i in tqdm(range(len(file_names)), ascii=True, desc="Processing images"):
        img = cv2.imread(file_names[i])
        # ################
        # ### UNDISTORT IMAGE
        # ################
        # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        # # # crop the image
        # x, y, w, h = roi
        # img = dst[y:y+h, x:x+w]
        # cv2.imwrite(f"tmp/{i}.jpg", img)

        ################
        ### BIN DETECTION AND CROP-TRANSFORM
        ################
        img = bin_crop(img)

        if isinstance(img, bool):
            continue

        ################
        ### MASK-RCNN - INSTANCE SEGMENTATION
        ################

        r = segmentation([img], i, True)[0]

        ################
        ### FILTER MASKS AND GET APPROX SIZES IN PIXEL
        ################

        sizes = get_ellipses_sizes(r, img)

        ################
        ### ROUGH WEIGHT ESTIMATE
        ################
        classes += converter(dimensions_px=sizes, bin_idx=i, specific_weight=0.54473, bias=0.8, logs=True )
        classes1 += converter(dimensions_px=sizes, bin_idx=i, specific_weight=0.57217, bias=0.8, logs=True )

    real_data_percentage = [
        3.35,
        13.78,
        10.39 + 1.99,
        11.99 + 1.44,
        13.27,
        12.96 + 1.17,
        11.64 + 0.34,
        4.72,
        3.17,
        3.05,
        6.74,
    ]
    labels = [
        "150+",
        "125/150",
        "115/125",
        "105/115",
        "95/105",
        "85/95",
        "75/85",
        "70/75",
        "65/70",
        "60/65",
        "industry",
    ]
    # Plotting

    # Calculate average percentages for classes and classes1
    avg_percentages_classes = (classes / classes.sum()) * 100
    avg_percentages_classes1 = (classes1 / classes1.sum()) * 100

    # Calculate average between classes and classes1
    average_values = np.mean(np.array([classes, classes1]), axis=0)

    # Plotting
    percentages = (average_values / average_values.sum()) * 100

    colors = plt.cm.Paired(range(2))
    colors1 = plt.cm.Set2(range(2))

    plt.figure(figsize=(12, 6), dpi=300)

    plt.bar(labels, percentages, color=colors[0], label="Avg. weight %", alpha=0.7)

    plt.scatter(
        labels, real_data_percentage, color=colors[1], label="Reference data", zorder=5
    )

    plt.scatter(
        labels,
        avg_percentages_classes,
        color=colors1[0],
        marker="_",
        s=500,
        label="Density: 544.73",
    )
    plt.scatter(
        labels,
        avg_percentages_classes1,
        color=colors1[1],
        marker="_",
        s=500,
        label="Density: 572.17",
    )

    plt.legend()
    plt.xlabel("Weight Classes [g]")
    plt.ylabel("Class percentage")
    plt.title("Distribution of Fruits by Weight Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("../final_plot.png", dpi=300)
    plt.show()
