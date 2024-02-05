"""
Classic approach to camera calibration with OpenCV

Source: [OpenCV Docs](https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html)
"""

import cv2
import numpy as np
import glob
import os
import logging


if __name__ == "__main__":
    PARAMETERS_PATH = os.path.abspath("./src/calibration")
    CAMERA_IMAGES_PATH = os.path.join(
        os.path.abspath("."), "datasets", "camera_calibration"
    )
else:
    PARAMETERS_PATH = os.path.abspath("src/calibration")
    CAMERA_IMAGES_PATH = os.path.abspath("datasets/camera_calibration")


def get_parameters():
    """
    Verify if the parameter file (`calibration_parameters.npz`) already exists.

    If it exists it returns its values, otherwise they are calculated.

    Return:  mtx, dist, newcameramtx, roi
    """

    if os.path.exists(os.path.join(PARAMETERS_PATH, "calibration_parameters.npz")):
        logging.info("Found parameters")
        values = np.load(os.path.join(PARAMETERS_PATH, "calibration_parameters.npz"))
        mtx, dist, newcameramtx, roi = (values["mtx"], values["dist"], values["newcameramtx"], values["roi"])
    else:
        from tqdm import tqdm

        logging.info("Parameters not found, calculating...")
        # #####
        # ### Calculate parameters
        # #####

        PATTERN_SIZE = (11, 17)

        # Termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (7,5,0)
        objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0 : PATTERN_SIZE[0], 0 : PATTERN_SIZE[1]].T.reshape(
            -1, 2
        )

        # Arrays to store object points and image points from all the images.
        obj_points = []  # 3d point in real world space
        img_points = []  # 2d points in image plane.

        calibration_images = glob.glob(os.path.join(CAMERA_IMAGES_PATH, "*.jpg"))

        h, w, gs = None, None, None

        for fname in tqdm(calibration_images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if h is None:
                h, w = img.shape[:2]
                gs = gray.shape[::-1]

            # Find chessboard corners
            ret, corners = cv2.findChessboardCornersSB(gray, PATTERN_SIZE)

            if ret:
                obj_points.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners2)

        # Check if there are enough calibration images
        if len(obj_points) < 5:
            raise ValueError(
                "Insufficient calibration images. At least 5 images are required."
            )

        # Perform camera calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, gs, None, None
        )

        # Get optimal new camera matrix and region of interest (ROI)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # Save calibration parameters
        np.savez(
            os.path.join(PARAMETERS_PATH, "calibration_parameters.npz"),
            mtx=mtx,
            dist=dist,
            newcameramtx=newcameramtx,
            roi=roi,
        )

    return  mtx, dist, newcameramtx, roi


if __name__ == "__main__":
    mtx, roi, newcameramtx = get_parameters()

    # Just remember that then you have to use cv2.undistort(img, mtx, dist, None, mtx)
    # undistort
    # dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    # cv.imwrite('calibresult.png', dst)
