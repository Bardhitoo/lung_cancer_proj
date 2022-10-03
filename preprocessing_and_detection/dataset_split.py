import os
import pandas as pd
import matplotlib
import random
import pickle

matplotlib.use('TkAgg')  # TODO: Experiment this on another env. of conda
import matplotlib.pyplot as plt
import SimpleITK as sitk

from sklearn.model_selection import train_test_split

SCAN_PATH = "../data/subset0"
CANDIDATE_FILE_PATH = "../data/candidates.csv"


def main():
    def filter_no_downloads(df, path):
        files = [file[:-4] for file in os.listdir(path) if file.endswith(".mhd")]
        my_df = pd.DataFrame(columns=df.columns)
        for file in files:
            if sum(df["seriesuid"] == file) > 0:
                my_df = pd.concat([my_df, df[df["seriesuid"] == file]], axis=0)

        return my_df

    total_candidates = pd.read_csv(CANDIDATE_FILE_PATH)

    # Filter out instances that I don't have on my local machine
    my_candidates = filter_no_downloads(total_candidates, SCAN_PATH)

    # Get positive and negative instances
    positive_cases = my_candidates[my_candidates["class"] == 1]
    negative_cases = my_candidates[my_candidates["class"] == 0]

    # Balance the positive to negative classes
    neg_idx = random.choices(negative_cases.index, k=3 * len(positive_cases))
    negative_cases = negative_cases.loc[neg_idx]

    my_candidates_df_idx = [list(positive_cases.index) + list(negative_cases.index)][0]
    my_candidates_df = total_candidates.loc[my_candidates_df_idx]

    X = my_candidates.iloc[:, :-1]
    y = my_candidates.iloc[:, -1]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    split_dataset = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

    with open(f'../data/split_dataset.pickle', 'wb') as ds:
        pickle.dump(split_dataset, ds, protocol=pickle.HIGHEST_PROTOCOL)

    # paths = [file for file in os.listdir(SCAN_PATH) if file.endswith(".mhd")]
    # s = sitk.ReadImage(os.path.join(SCAN_PATH, paths[2]))
    # image = sitk.GetArrayFromImage(s)
    # # df[(df.seriesuid == paths[2][:-4]) &  (df["class"] == 1)]


if __name__ == "__main__":
    main()
