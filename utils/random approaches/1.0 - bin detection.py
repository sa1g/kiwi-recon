import argparse

from os import listdir, makedirs
from os.path import isfile, join, isdir, exists

import cv2
import numpy as np
from matplotlib import pyplot as plt
import logging
from typing import List
from tqdm import tqdm

def dir_path(string):
    if isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def set_parsers():
    """Setup program parsers.

    Returns:
        arguments of the program.
    """
    parser = argparse.ArgumentParser(
        description="Process a bunch of images by cropping them on the bin fruit (if detected)"
    )
    parser.add_argument(
        "--in-path",
        type=dir_path,
        default="./dataset/raw-selected",
        help="Path for the input directory",
    )
    
    parser.add_argument(
        "--out-path",
        type=dir_path,
        default="./dataset",
        help="Path for the output directory location",
    )
    return parser.parse_args()

def check_filenames(in_path) -> List[str]:
    """Verify if 'args.in_path' exists and keeps only file names with
    `PNG`, `JPEG` and `JPG` extension.

    Args:
        in_path: Input path from args.

    Returns:
        valid_file_names: list of valid file names.
    """
    file_names = [f for f in listdir(in_path) if isfile(join(in_path, f))]

    # Filter out files without .PNG, .JPEG, or .JPG extension
    valid_file_names = [
        f for f in file_names if f.lower().endswith((".png", ".jpeg", ".jpg"))
    ]

    invalid_file_names = set(file_names) - set(valid_file_names)
    if invalid_file_names:
        logging.warning(
            f"The following files have invalid extensions and will be ignored: {invalid_file_names}"
        )

    return valid_file_names

def create_output_dir(base_path: str) -> str:
    """Create an output directory. If the directory already exists, append a suffix.

    Args:
        base_path (str): Base path for the output directory.

    Returns:
        output_dir (str): Path to the created or modified output directory.
    """
    output_dir = base_path

    # Check if the directory already exists
    if exists(output_dir):
        print("CODIO")
        # Append a suffix to the directory name
        index = 1
        while exists(output_dir):
            output_dir = f"{base_path}-{index}"
            index += 1

    # Create the directory
    makedirs(output_dir)

    return output_dir

def __order_points(points) -> np.array:
    """Utility function - order coordinates tuple by position in by quadrant position.

    Args:
        points (np.array): Unordered array of points.

    Returns:
        ordered points: Ordered array of points.
    """
    # Convert the input points to a NumPy array
    points = np.array(points)

    # Initialize an array to store the ordered points
    ordered_points = np.zeros_like(points)

    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Calculate the angles between the centroid and each point
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    # Sort the points based on angles
    ordered_indices = np.argsort(angles)
    ordered_points = points[ordered_indices]

    return ordered_points


def bin_crop(in_path: str, out_path:str, file_names: List[str]) -> None:
    """Crop each `file_names` image on the found fruit bin.

    Args:
        in_path (str): Input path from args.
        out_path (str): Output path from args.
        file_names (List[str]): list of file names.
    """
    # Create output dir if not-existent, otherwise create another one with different name
    output_dir = create_output_dir(out_path)
    for count, file in tqdm(enumerate(file_names)):
        try:
            # Read the image
            image = cv2.imread(f"{in_path}/{file}")
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_color = (79, 0, 179)
            upper_color = (128, 255, 255)

            # Create a binary mask based on color segmentation
            thresh = cv2.inRange(hsv, lower_color, upper_color)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            dilated = cv2.dilate(morph, kernel, iterations=2)

            # # Apply GaussianBlur to reduce noise
            blurred = cv2.GaussianBlur(dilated, (5, 5), 0)

            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 100, 150)

            # Combine color and edge masks - looks like it's adding noise
            combined_mask = cv2.bitwise_or(dilated, edges)

            # # Find contours in the combined mask
            contours, _ = cv2.findContours(
                combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Find the contour with the maximum area (assumed to be the bin)
            max_contour = max(contours, key=cv2.contourArea)

            epsilon = 0.02 * cv2.arcLength(max_contour, True)
            approx_contour = cv2.approxPolyDP(max_contour, epsilon, True)

            red_box_min_side = np.int32(
                np.abs(approx_contour[:, 0, 0] - approx_contour[:, 0, 1]).mean()
            )

            # Define the target points for perspective correction
            target_points = np.array(
                [
                    [red_box_min_side, red_box_min_side],
                    [0, red_box_min_side],
                    [red_box_min_side, 0],
                    [0, 0],
                ],
                dtype=np.float32,
            )

            # Order the points
            approx_ordered = __order_points(approx_contour.reshape(-1, 2)).reshape(-1, 1, 2)
            target_points_ordered = __order_points(target_points).reshape(-1, 1, 2)

            # Calculate the perspective transformation matrix
            perspective_matrix = cv2.getPerspectiveTransform(
                approx_ordered.reshape(4, 2).astype(np.float32),
                target_points_ordered.reshape(4, 2),
            )

            dst = cv2.warpPerspective(
                image, perspective_matrix, (red_box_min_side, red_box_min_side)
            )
            
            cv2.imwrite(f"{output_dir}/{count}.png", dst)
        except ValueError as e:
            logging.error(f"Image {file} | {e}")
        

if __name__ == "__main__":
    args = set_parsers()

    in_path = args.in_path
    out_path = f"{args.out_path}/selected"

    valid_file_names = check_filenames(in_path)

    ### Process images
    # Crop on the bin
    bin_crop(in_path, out_path, valid_file_names)
