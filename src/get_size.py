import cv2
import numpy as np


# def __group_contours(contours, max_distance):
#     """
#     Group contours when they are close.

#     Args:
#         contours (list): contours given by cv2
#         max_distance (int): max distance in px

#     Returns:
#         contours clusters (list)
#     """
#     grouped_contours = []
#     used = set()

#     for i, contour1 in enumerate(contours):
#         if i not in used:
#             group = [contour1]
#             used.add(i)

#             for j, contour2 in enumerate(contours[i + 1 :]):
#                 j = j + i + 1
#                 if j not in used:
#                     M1 = cv2.moments(contour1)
#                     centroid1 = (
#                         int(M1["m10"] / M1["m00"]),
#                         int(M1["m01"] / M1["m00"]),
#                     )

#                     M2 = cv2.moments(contour2)
#                     centroid2 = (
#                         int(M2["m10"] / M2["m00"]),
#                         int(M2["m01"] / M2["m00"]),
#                     )

#                     distance = np.linalg.norm(np.array(centroid1) - np.array(centroid2))

#                     if distance < max_distance:
#                         group.append(contour2)
#                         used.add(j)

#             grouped_contours.append(group)

#     return grouped_contours


def __group_contours(contours, max_distance):
    """
    Group contours when they are close.

    Args:
        contours (list): contours given by cv2
        max_distance (int): max distance in px

    Returns:
        contours clusters (list)
    """
    grouped_contours = []
    used = set()

    for i, contour1 in enumerate(contours):
        if i not in used:
            group = [contour1]
            used.add(i)

            M1 = cv2.moments(contour1)
            area1 = M1["m00"]

            if area1 > 0:  # Check if the area is non-zero
                centroid1 = (
                    int(M1["m10"] / area1),
                    int(M1["m01"] / area1),
                )

                for j, contour2 in enumerate(contours[i + 1 :]):
                    j = j + i + 1
                    if j not in used:
                        M2 = cv2.moments(contour2)
                        area2 = M2["m00"]

                        if area2 > 0:  # Check if the area is non-zero
                            centroid2 = (
                                int(M2["m10"] / area2),
                                int(M2["m01"] / area2),
                            )

                            distance = np.linalg.norm(
                                np.array(centroid1) - np.array(centroid2)
                            )

                            if distance < max_distance:
                                group.append(contour2)
                                used.add(j)

                grouped_contours.append(group)

    return grouped_contours


def get_ellipses_sizes(r: list, image):
    """
    This function filters out kiwis masks and returns the estimated sizes in pixels.

    Workflow:


    Args:
        r: Mask-RCNN single image detected results for kiwi-fruit.

    Returns:
        # List(Tuple(height, width)).
        {"img_shape": image.shape, "dimensions": dimensions}
    """

    mask = r["masks"]
    scores = r["scores"]
    roi = r["rois"]

    # For some reasons the area difference even when they are very similar is quit huge, for this reason I'm applying this multiplier. I'm sure there's a reason for this issue, I just can't find it right now. I'm not using cv2.contourArea(polygons[0]) as it would add an additional stage of error.
    AREA_MULTIPLIER = 4
    SCORE_THRESHOLD = 0.7
    AREA_THRESHOLD = 0.005
    ELLIPSE_THRESHOLD = 0.05
    IMAGE_AREA = image.shape[0] * image.shape[1]

    dimensions = []
    for i in range(mask.shape[-1]):
        # Select the current mask
        mask_single = mask[:, :, i]

        # Convert boolean mask to uint8
        mask_single = mask_single.astype(np.uint8) * 255
        area = np.sum(mask_single) * AREA_MULTIPLIER

        # Find contours using cv2.findContours
        contours, _ = cv2.findContours(
            mask_single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if scores[i] < SCORE_THRESHOLD:
            continue

        # Skip areas that are too small
        if area < AREA_THRESHOLD * IMAGE_AREA:
            continue

        # Convert contours to polygons
        polygons = [
            cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
            for contour in contours
        ]

        # If multiply polygons are found skip them
        if len(polygons) != 1:
            continue

        hull = cv2.convexHull(polygons[0])

        # Calculate ellipse
        ellipse_int = cv2.fitEllipse(hull)
        ellipse_ext = cv2.fitEllipse(polygons[0])

        major_axis_ext, minor_axis_ext = ellipse_ext[1]
        major_axis_int, minor_axis_int = ellipse_int[1]

        # Calculate the area of the ellipse
        ellipse_int_area = np.pi * major_axis_int * minor_axis_int
        ellipse_ext_area = np.pi * major_axis_ext * minor_axis_ext

        if (
            np.abs(ellipse_ext_area - ellipse_int_area)
            > np.mean([ellipse_ext_area, ellipse_int_area]) * ELLIPSE_THRESHOLD
        ):
            continue

        ### Find stuff inside the kiwi
        normalizedImg = np.zeros((800, 800))
        normalizedImg = cv2.normalize(image, normalizedImg, 0, 255, cv2.NORM_MINMAX)
        # normalizedImg = image.copy()

        # Crop the image around the fruit
        c = cv2.bitwise_and(
            normalizedImg, cv2.cvtColor(mask_single, cv2.COLOR_GRAY2BGR)
        )[roi[i][0] - 10 : roi[i][2] + 20, roi[i][1] - 10 : roi[i][3] + 10]

        gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        blurred = cv2.GaussianBlur(morph, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200)

        internal_contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        internal_contours = __group_contours(internal_contours, 5)

        if len(internal_contours) > 1:
            continue

        # Memorize avg height and width (from the 2 ellipses)
        mean_1, mean_2 = np.mean([major_axis_ext, major_axis_int]), np.mean(
            [minor_axis_ext, minor_axis_int]
        )

        ##### Save Dimensions
        avg_height, avg_width = np.max([mean_1, mean_2]), np.min([mean_1, mean_2])
        dimensions.append((avg_height, avg_width))

    return {"img_shape": image.shape, "dimensions": dimensions}


if __name__ == "__main__":
    import pickle
    import os

    with open(
        os.path.join(
            os.path.abspath(""),
            "src",
            "mask_rcnn",
            "mrcnn_single_out.pickle",
        ),
        "rb",
    ) as f:
        r = pickle.load(f)

    image = cv2.imread(
        os.path.join(os.path.abspath(""), "dataset", "selected", "0.png")
    )

    import matplotlib.pyplot as plt

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.savefig("a.png")

    print(get_ellipses_sizes(r, image))
