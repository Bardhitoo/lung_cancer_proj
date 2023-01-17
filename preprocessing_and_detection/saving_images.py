import os
import pickle
import SimpleITK as sitk
from utils.my_utils import *
import cv2

import matplotlib
import pandas as pd

matplotlib.use('TkAgg')
from preprocessing_and_detection.utils.my_utils import *

if not os.path.exists(SAVE_IMG):
    os.mkdir(SAVE_IMG)

if not os.path.exists(SAVE_IMG + "train"):
    os.mkdir(SAVE_IMG + "train")

if not os.path.exists(SAVE_IMG + "test"):
    os.mkdir(SAVE_IMG + "test")


def main():
    with open(SAVE_SPLIT_PATH, "rb") as input_file:
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

        for index, sample in samples.iterrows():
            sample_name = sample["seriesuid"]
            sample_objt = sitk.ReadImage(sample["path"])

            # === Get only CT scans whose resolution in z-axis is greater than 2 ===
            # From the exploratory analysis there were only a number of participants
            # Who had 2.5mm resolution across the z-axis
            # if sample_objt.GetSpacing()[2] <= 2:
            #     continue

            counter += 1
            print(f"Processing: {dataset_name} - {counter}")
            if sample_name in seen:
                seen[sample_name] = seen[sample_name] + 1
            else:
                seen[sample_name] = 0
            image_path = f"{SAVE_IMG}/{dataset_name}/{sample['seriesuid']}_{seen[sample_name]}.jpg"

            image = sitk.GetArrayFromImage(sample_objt)

            x_, y_, z = get_image_coords(sample_objt, sample)

            dist = 1  # round(2. / sample_objt.GetSpacing()[2])
            image_slice_m1 = image[round(z) - dist, :, :]
            image_slice_0 = image[round(z), :, :]
            image_slice_p1 = image[round(z) + dist, :, :]

            # viz_bounding_box(image_slice_m1, x_, y_, width=40)
            # viz_bounding_box(image_slice_0, x_, y_, width=40)
            # viz_bounding_box(image_slice_p1, x_, y_, width=40)
            # viz_bounding_box(image_normalized, x_, y_, labels[index], width=40)
            image_m1_normalized = normalise_planes(image_slice_m1)
            image_0_normalized = normalise_planes(image_slice_0)
            image_p1_normalized = normalise_planes(image_slice_p1)

            # Stack images
            image_normalized = cv2.merge((image_m1_normalized, image_0_normalized, image_p1_normalized))
            cv2.imwrite(image_path, image_normalized * 255)

            df.loc[len(df.index)] = [sample_name, x_, y_, z, labels[index], sample["path"]]
        df.to_csv(f"./processed_data/{dataset_name}_{EXPERIMENT_NAME}_transformed_coords.csv", index=False)


if __name__ == "__main__":
    main()
