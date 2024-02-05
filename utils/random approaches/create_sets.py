"""
This script will split files from `Fruit-Images-Dataset`, `maskrsnn-dataset-augmented` and `random` into the 3 types of experiments:
- `exp_1`: only `maskrcnn-dataset-augmented`
- `exp_2`: divided in 2 phases
    - `pt_1`: `Fruit-Images-Dataset`, `random`
    - `pt_2`: `maskrcnn-dataset-augmented`
- `exp_3`: `Fruit-Images-Dataset`, `maskrsnn-dataset-augmented`, `random`

All experiments have 3 sub directories:
- train (80%)
- test  (10%)
- val   (10%)
"""

import json
import os
import shutil
import numpy as np
from typing import Tuple


def split_dataset(via_export_json: dict = dict) -> Tuple[dict, dict, dict]:
    """
    Split the data in three categories: train (80%), test (10%), val (10%).

    Args:
        via_export_json: VGG Image Annotator json output

    Returns:
        train, test, val
    """

    train, test, val = {}, {}, {}
    for key, value in via_export_json.items():
        choice = np.random.randint(0, 10)

        if choice == 0:
            val[key] = value
        elif choice == 1:
            test[key] = value
        else:
            train[key] = value

    return train, test, val


def copyfile_add_to_structure(
    IN_PATH: str, OUT_PATH: str, output_json: dict, new_data: dict
) -> dict:
    """
    It adds the new datastructure infos inside `output_json`, copies the file to the specified output dir.
    """
    for t1, t2 in zip(output_json.values(), new_data.values()):
        t1.update(t2)

    for tt, data in new_data.items():
        # tt is train, test, val
        for img in data.values():
            in_dir = os.path.join(IN_PATH, img["filename"])
            out_dir = os.path.join(OUT_PATH, tt, img["filename"])

            shutil.copyfile(in_dir, out_dir)

    return output_json


def save_json(output_json: dict, save_path: str):
    """
    Save output_json automatically into the three different categories:
    - train
    - test
    - val

    Args:
        output_json: output ds
        save_path: without file name
    """

    for key, value in output_json.items():
        with open(
            os.path.join(save_path, key, "via_export_json.json"), "w+", encoding="utf-8"
        ) as file:
            json.dump(value, file)

        print(f"{key} -- {len(value)+1}")


def create_dirs(structure: dict, abs_path: str) -> None:
    """
    WARNING: recursive!

    Create directories given the specified experiments structure.
    It will create at least 4 dirs per each experiment:
    - experiment
        - train
        - test
        - val

    Args:
        structure: exp structure
        path: abs path
    """
    for key, value in structure.items():
        hst = os.path.join(abs_path, key)
        os.makedirs(hst, exist_ok=True)

        if isinstance(value, dict):
            create_dirs(value, hst)
        else:
            os.makedirs(os.path.join(hst, "test"), exist_ok=True)
            os.makedirs(os.path.join(hst, "train"), exist_ok=True)
            os.makedirs(os.path.join(hst, "val"), exist_ok=True)


if __name__ == "__main__":
    PATH = os.path.join(os.path.abspath("."), "dataset")

    """
    1. Create per each data source the three subsets (as dict)
    2. On each experiment put data from the correct sources both imgs and json 
    """

    ##############################
    # Define sources
    ##############################

    PATH_DS = {
        "fid_ds": os.path.join(PATH, "storage", "Fruit-Images-Dataset", "Training"),
        # or augmented
        "mda_ds": os.path.join(PATH, "storage", "maskrcnn-dataset"),
        "rnd_ds": os.path.join(PATH, "storage", "random"),
    }

    ##############################
    # Split sets
    ##############################
    ds = {
        "fid_ds": {},
        "mda_ds": {},
        "rnd_ds": {},
    }

    for path, dd in zip(PATH_DS.values(), ds.values()):
        with open(
            os.path.join(path, "via_export_json.json"), "r", encoding="utf-8"
        ) as file:
            json_data = json.load(file)

        dd["train"], dd["test"], dd["val"] = split_dataset(json_data)

    ##############################
    # Experiments config
    ##############################

    EXPS = {
        "exp_1": "mda_ds",
        "exp_2": {"pt_1": ["fid_ds", "rnd_ds"], "pt_2": "mda_ds"},
        "exp_3": ["mda_ds", "fid_ds", "rnd_ds"],
    }

    ##############################
    # Create directories
    ##############################

    create_dirs(EXPS, PATH)

    ##############################
    # Create experiments
    # This should be done recursively
    ##############################

    for key, value in EXPS.items():
        print(key)
        # For each experiment define a new output file
        output_json = {
            "train": {},
            "test": {},
            "val": {},
        }

        if isinstance(value, dict):
            for k, v in value.items():
                print(k)
                output_json = {
                    "train": {},
                    "test": {},
                    "val": {},
                }

                if isinstance(v, list):
                    for df in v:
                        output_json = copyfile_add_to_structure(
                            PATH_DS[df], os.path.join(PATH, key, k), output_json, ds[df]
                        )
                else:
                    output_json = copyfile_add_to_structure(
                        PATH_DS[v], os.path.join(PATH, key, k), output_json, ds[v]
                    )

                save_json(output_json, os.path.join(PATH, key, k))
            continue

        elif isinstance(value, list):
            for df in value:
                output_json = copyfile_add_to_structure(
                    PATH_DS[df], os.path.join(PATH, key), output_json, ds[df]
                )

        else:
            output_json = copyfile_add_to_structure(
                PATH_DS[value], os.path.join(PATH, key), output_json, ds[value]
            )

        save_json(output_json, os.path.join(PATH, key))
