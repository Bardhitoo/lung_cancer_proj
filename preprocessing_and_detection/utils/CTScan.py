import os
import matplotlib.pyplot as plt

from preprocessing_and_detection.utils.my_utils import normalise_planes
import SimpleITK as sitk


class CTScan:
    def __init__(self, path):
        assert os.path.exists(path)  # f"Path does not exist! {path}"
        self.path = path
        self.s = None
        self.image = None
        self.shape = -1

    def __str__(self):
        return f"{self.path}"

    def load_image(self):
        self.s = sitk.ReadImage(self.path)
        self.image = sitk.GetArrayFromImage(self.s)
        self.shape = self.image.shape

    def get_slice(self, slice_num):
        if not self.s:
            self.load_image()
        normalized_slice = normalise_planes(self.image[slice_num, :, :])
        return normalized_slice
