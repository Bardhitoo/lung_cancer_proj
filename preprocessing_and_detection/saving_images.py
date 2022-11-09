import os
import pickle
import SimpleITK as sitk
import numpy as np
import cv2

import matplotlib
import pandas as pd
import tensorflow as tf

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from my_records import *

if not os.path.exists(SAVE_IMG):
    os.mkdir(SAVE_IMG)

if not os.path.exists(SAVE_IMG + "train"):
    os.mkdir(SAVE_IMG + "train")

if not os.path.exists(SAVE_IMG + "test"):
    os.mkdir(SAVE_IMG + "test")


def normalizePlanes(npzarray):
    """
    Copied from SITK tutorial converting Houndsunits to grayscale units
    """
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.
    npzarray[npzarray < 0] = 0.
    return npzarray


def viz_bounding_box(image_slice, x, y, width=40):
    # Display the image
    plt.figure()
    plt.imshow(image_slice)
    ax = plt.gca()
    # Create a Rectangle patch
    rect = patches.Rectangle((x - width / 2, y - width / 2), width, width, linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.show()


def get_image_coords(s, row):
    real_world_coords = row[1:].tolist()
    origin = s.GetOrigin()
    res = s.GetSpacing()

    if res[2] != 0.625:
        raiseValueError = 1
        # raise ValueError("Inconsistent spacing")

    x, y, z = [np.abs(real_world_coords[j] - origin[j]) / res[j] for j in range(len(real_world_coords))]
    return x, y, z


def main():
    with open(TRAIN_DATA_PATH, "rb") as input_file:
        train_test_split_dict = pickle.load(input_file)

    dataset = train_test_split_dict["dataset"]
    X_train = train_test_split_dict["X_train"]
    y_train = train_test_split_dict["y_train"]
    X_test = train_test_split_dict["X_test"]
    y_test = train_test_split_dict["y_test"]
    for dataset_name, samples, labels in zip(["train", "test"], [X_train, X_test], [y_train, y_test]):
        df = pd.DataFrame(columns=dataset.columns)
        seen = {}
        counter = 0

        for index, row in samples.iterrows():
            sample_name = row["seriesuid"]
            s = sitk.ReadImage(os.path.join(SCAN_PATH, sample_name + ".mhd"))

            # === Get only CT scans whose resolution in z-axis is greater than 2 ===
            # From the exploratory analysis there were only a number of participants
            # Who had 2.5mm resolution across the z-axis
            if s.GetSpacing()[2] <= 2:
                continue

            counter += 1
            print(f"Processing: {dataset_name} - {counter}")
            if sample_name in seen:
                seen[sample_name] = seen[sample_name] + 1
            else:
                seen[sample_name] = 0
            image_path = f"{SAVE_IMG}/{dataset_name}/{row['seriesuid']}_{seen[sample_name]}.jpg"

            image = sitk.GetArrayFromImage(s)

            x_, y_, z = get_image_coords(s, row)

            image_slice_m1 = image[int(z) - 1, :, :]
            image_slice_0 = image[int(z), :, :]
            image_slice_p1 = image[int(z) + 1, :, :]

            # viz_bounding_box(image_slice_m1, x_, y_, width=40)
            # viz_bounding_box(image_slice_0, x_, y_, width=40)
            # viz_bounding_box(image_slice_p1, x_, y_, width=40)

            image_m1_normalized = normalizePlanes(image_slice_m1)
            image_0_normalized = normalizePlanes(image_slice_0)
            image_p1_normalized = normalizePlanes(image_slice_p1)

            # Stack images
            image_normalized = cv2.merge((image_m1_normalized, image_0_normalized, image_p1_normalized))
            cv2.imwrite(image_path, image_normalized * 255)

            df.loc[len(df.index)] = [sample_name, x_, y_, z, labels[index]]
        df.to_csv(f"processed_data/{dataset_name}_{EXPERIMENT_NAME}_transformed_coords.csv", index=False)


if __name__ == "__main__":
    main()
