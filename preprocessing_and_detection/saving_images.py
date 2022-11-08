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

DATA = "../data/"
SCAN_PATH = DATA + "subset0"
TRAIN_DATA_PATH = DATA + "split_dataset.pickle"
SAVE_IMG = DATA + "subset0_3D_2_train_test_split/"

if not os.path.exists(SAVE_IMG + "train"):
    os.mkdir(SAVE_IMG + "train")

if not os.path.exists(SAVE_IMG + "test"):
    os.mkdir(SAVE_IMG + "test")


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


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


def viz_bounding_box(image_slice, x, y, width=30):
    # Display the image
    plt.figure()
    plt.imshow(image_slice)
    ax = plt.gca()
    # Create a Rectangle patch
    rect = patches.Rectangle((x - width / 2, y - width / 2), width, width, linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.show()


def main():
    with open(TRAIN_DATA_PATH, "rb") as input_file:
        train_test_split_dict = pickle.load(input_file)

    dataset = train_test_split_dict["dataset"]
    X_train = train_test_split_dict["X_train"]
    y_train = train_test_split_dict["y_train"]
    X_test = train_test_split_dict["X_test"]
    y_test = train_test_split_dict["y_test"]

    for dataset, dataset_name in zip([X_train, X_test], ["train", "test"]):
        df = pd.DataFrame(columns=dataset.columns)
        seen = {}
        counter = 0

        for index, row in dataset.iterrows():
            counter += 1
            print(f"Processing: {dataset_name} - {counter} / {len(dataset) - 1}")
            sample_name = row["seriesuid"]
            s = sitk.ReadImage(os.path.join(SCAN_PATH, sample_name + ".mhd"))

            if sample_name in seen:
                seen[sample_name] = seen[sample_name] + 1
            else:
                seen[sample_name] = 0
            image_path = f"{SAVE_IMG}/{dataset_name}/{row['seriesuid']}_{seen[sample_name]}.jpg"

            image = sitk.GetArrayFromImage(s)

            x_, y_, z = get_image_coords(s, row)

            image_slice_m1 = image[int(z) - 1, :, :]
            image_slice_0 = image[int(z)     , :, :]
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

            df.loc[len(df.index)] = [sample_name, x_, y_, z]
        df.to_csv(f"processed_data/{dataset_name}_3D_2_transformed_coords.csv")


def get_image_coords(s, row):
    real_world_coords = row[1:].tolist()
    origin = s.GetOrigin()
    res = s.GetSpacing()

    if res[2] != 0.625:
        raiseValueError = 1
        # raise ValueError("Inconsistent spacing")

    x, y, z = [np.abs(real_world_coords[j] - origin[j]) / res[j] for j in range(len(real_world_coords))]
    return x, y, z


if __name__ == "__main__":
    main()