import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle

DATA = "../data/"
# SCAN_PATH = DATA + "subset0"
CANDIDATE_FILE_PATH = DATA + "candidates_V2.csv"
SAVE_SPLIT_PATH = DATA + "split_dataset_0123.pickle"
EXPERIMENT_NAME = "full_subset0123"
SAVE_IMG = DATA + f"{EXPERIMENT_NAME}_train_test_split/"
PATH_TO_MODEL_DIR = ""


@dataclass
class Prediction:
    CTScan_patient_id: str
    CTScan_path: str
    saved_predictions: Dict = field(default_factory=dict)

    def add(self, slice_num, seg_contour, bbox):
        self.saved_predictions[slice_num] = {"seg_contour": seg_contour, "bbox": bbox}

    def to_pickle(self, filepath):
        self._flag_preds()
        with open(filepath, 'wb') as handle:
            pickle.dump(dataclasses.asdict(self), handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, filepath):
        with open(filepath, 'rb') as picked_filed:
            model_result = pickle.load(picked_filed)
        return cls(**model_result)

    def _flag_preds(self):
        """
        Flags out individual slices without adjacent neighbors
        :return:
        """

        idx = 0
        pred_slice_list = list(self.saved_predictions.keys())
        while idx < len(pred_slice_list):
            slice_idx = pred_slice_list[idx]

            # If they are adjacent - continue to create streak of adjacent slices
            blob_len = 1

            while idx < (len(pred_slice_list) - 1) and slice_idx + 1 == pred_slice_list[idx + 1]:
                self.saved_predictions[slice_idx]["flag"] = 0
                blob_len += 1
                slice_idx = pred_slice_list[idx + 1]
                idx += 1

            # edge case = if there's a single slice, flag it
            if blob_len > 1:
                self.saved_predictions[slice_idx]["flag"] = 0
            else:
                self.saved_predictions[slice_idx]["flag"] = 1

            idx += 1


def viz_bounding_box(image_slice, x, y, label, width=40):
    # Display the image
    plt.figure()
    plt.imshow(image_slice)
    ax = plt.gca()
    # Create a Rectangle patch
    rect = patches.Rectangle((x - (width / 2), y - (width / 2)), width, width, linewidth=1, edgecolor='r',
                             facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.text(x - (width / 2), y - (width / 2), label, color='yellow', weight='bold', fontsize=15)
    plt.show()


def get_image_coords(s, sample):
    real_world_coords = sample[["coordX", "coordY", "coordZ"]].tolist()
    origin = s.GetOrigin()
    res = s.GetSpacing()

    x, y, z = [np.abs(real_world_coords[j] - origin[j]) / res[j] for j in range(len(real_world_coords))]
    return x, y, z


def normalise_planes(npzarray):
    """
    Copied from SITK tutorial converting Houndsunits to grayscale units
    """
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.
    npzarray[npzarray < 0] = 0.
    return npzarray
