import os
import pandas as pd
import matplotlib
import random
import pickle

matplotlib.use('TkAgg')  # TODO: Experiment this on another env. of conda

from sklearn.model_selection import train_test_split
from my_records import *

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

    #  TODO: Separate positive & negative cases by patient id
    # Dont want to have samples of patients split into train and test

    # Get positive and negative instances
    positive_cases = my_candidates[my_candidates["class"] == 1]
    negative_cases = my_candidates[my_candidates["class"] == 0]

    # Balance the positive to negative classes
    neg_idx = random.choices(negative_cases.index, k=3 * len(positive_cases))
    negative_cases = negative_cases.loc[neg_idx]

    my_candidates_df_idx = [list(positive_cases.index) + list(negative_cases.index)][0]
    my_candidates_df = total_candidates.loc[my_candidates_df_idx]

    X = my_candidates_df.iloc[:, :-1]
    y = my_candidates_df.iloc[:, -1]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    split_dataset = {"dataset": my_candidates_df, "X_train": X_train, "X_test": X_test, "y_train": y_train,
                     "y_test": y_test}

    with open(f'../data/split_dataset.pickle', 'wb') as output_file:
        pickle.dump(split_dataset, output_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
