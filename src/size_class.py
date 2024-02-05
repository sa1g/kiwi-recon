import pickle
import numpy as np

BIN_SIZE_CM = 114

def converter(dimensions_px:dict, bin_idx:int, specific_weight:float, bias:float=1.0, logs:bool=False):
    """Basic function that converts pixels to mm

    Args:
        dimensions_px: dict{'img_shape':[x,y,z],'dimensions':list(tuple)}
            img_shape is required as an `image.shape` with shape [x,x,n], x representing the axes of a squared image, n the dimensions (not used and considered).
        bin_idx: the index of the bin
    
    Returns:
        list[classes]
    """
    ## Convert dimensions in mm
    bin_px = dimensions_px["img_shape"][0]

    if logs:
        data = []
    classes = [0] * 11

    for dimension in dimensions_px["dimensions"]:
        height, width = (
            dimension[0] * BIN_SIZE_CM / bin_px,
            dimension[1] * BIN_SIZE_CM / bin_px,
        )

        ## Calculate volume edtimated as an elissoid with a=height, b=c=width
        ## Volume in mm^3
        volume_cm = (4 / 3) * np.pi * height * width / 2 * width / 2
        if logs:
            data.append([height,width,volume_cm])

        weight_est = volume_cm * specific_weight * bias

        weight_est = int(
            np.floor(weight_est) if weight_est < 0.5 else np.ceil(weight_est)
        )

        thresholds = [150, 125, 115, 105, 95, 85, 75, 70, 65, 60]
        for i, threshold in enumerate(thresholds):
            if weight_est > threshold:
                classes[i] += 1
                break
        else:
            classes[-1] += 1
    if logs:
        with open(f"logs/hwv-{bin_idx}.pickle", "wb+") as f:
            pickle.dump(data, f)

    return np.array(classes)
