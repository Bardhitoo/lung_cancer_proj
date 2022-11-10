import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

DATA = "../data/"
SCAN_PATH = DATA + "subset0"
TRAIN_DATA_PATH = DATA + "split_dataset.pickle"
CANDIDATE_FILE_PATH = DATA + "candidates_V2.csv"
DATA_PATH = DATA + "split_dataset.pickle"
EXPERIMENT_NAME = "full_subset0_1_3D_2mm"
SAVE_IMG = DATA + f"{EXPERIMENT_NAME}_train_test_split/"
PATH_TO_MODEL_DIR = ""


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

