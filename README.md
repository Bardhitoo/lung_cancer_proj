# lung_cancer_proj
This projects aims to explore and design an effective 3-step pipeline for lung cancer 1) detection, 2) segmentation, and 3) 3D reconstruction.

Why is this problem worth solving? 
* Problem
  * Solution
* Problem
  * Solution
* ...


# 1. Detection


### 1.0 Data  
In order to be able to give the data to detections algorithm, the data needs to be: processed, collected, standardized, and normalized.
This project uses a subset of data from Lung Image Database Consortium and Infectious Disease Research Institute ([LIDC/IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)) database,
specifically from the available online dataset of [LUNA16](https://luna16.grand-challenge.org).

This dataset consists of 888 CT scans with annotations (detection and segmentation) describing coordinates and ground truth labels.
The first step is to process the data and create an image database for training.

## 1.1 Preprocessing
### 1.1.1 Dataset processing
- Select patients with nodule >= 3mm
- Tackling the problem in 2D (image)
  - 100 positive instances (slices from 3D CT Scan of patience with    lung cancer)
  - 100 negative instances (slices from 3D CT Scan of patience without lung cancer)
  -> 3h

- Proposed processing steps:
  - Segment out lungs using classical CV (to reduce information space)
  - Have a nodule detector to find all potential nodules
  - Classify between benign and malignant cases

- Discussions:
  - Dataset is greatly imbalanced - 
