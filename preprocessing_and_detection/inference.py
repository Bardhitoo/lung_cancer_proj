import os
from preprocessing_and_detection.utils.my_utils import Prediction
import cv2
import time
import argparse
import numpy as np
import pylidc as pl
from scipy.ndimage.morphology import binary_fill_holes

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.detector import DetectorTF2
from utils.CTScan import CTScan


def normalizePlanes(npzarray, minHU=-1000., maxHU=400.):
    """
    Copied from SITK tutorial converting Houndsunits to grayscale units
    """

    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.
    npzarray[npzarray < 0] = 0.
    return npzarray


def consensus(img, slices_of_interest, percentage=0.5):
    filled_slices = np.zeros(img.shape[:2])
    for slice_of_interest in slices_of_interest:
        empty_img = np.zeros((img.shape[:2]))
        a = cv2.drawContours(empty_img, slice_of_interest[:, :2].reshape(-1, 1, 2), contourIdx=-1, color=1,
                             thickness=cv2.FILLED)
        filled_slices += binary_fill_holes(a)

    return filled_slices >= np.max(filled_slices) * percentage


def visualize_patch():
    patient_id = 'LIDC-IDRI-0078'
    scan_obj = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
    scan = scan_obj.to_volume()

    anns, slice_contours_matrix = concat_anns(scan_obj)

    slice_num = 0
    slice_contour_num = 0

    preds = Prediction('path_to_patient_id', patient_id)
    while slice_num < scan.shape[2] - 2:
        img = extract_img_from_scan(scan, slice_num)

        # are there two or more adjacent slices with cancer
        # if there are adjacent detection, create a streak

        # if the streak is longer than two
        # and if the box has some overlap than make consider a detection

        # detections contains cancer
        if slice_contour_num < len(slice_contours_matrix) and slice_num == slice_contours_matrix[slice_contour_num][0][
            2]:
            slices_of_interest = []

            # collect detections in the slices
            while slice_contour_num < len(slice_contours_matrix) and slice_num == \
                    slice_contours_matrix[slice_contour_num][0][2]:
                slices_of_interest.append(slice_contours_matrix[slice_contour_num])
                slice_contour_num += 1

            # if there are 2 or more detections, find consensus among those
            if len(slices_of_interest) >= 2:
                # handle multiple detections
                mask = consensus(img, slices_of_interest, percentage=0.5)
                slice_of_interest, _ = cv2.findContours(np.uint8(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            else:
                slice_of_interest = slices_of_interest
            slice_of_interest = slice_of_interest[0].reshape(-1, 2)

            # extract bounding box
            x_min, y_min = min(slice_of_interest[:, 1]), min(slice_of_interest[:, 0])
            x_max, y_max = max(slice_of_interest[:, 1]), max(slice_of_interest[:, 0])

            preds.add(slice_num, slice_of_interest, [x_min, y_min, x_max, y_max])
            # plot(arr[:, 1], arr[:, 0], '-r')
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
            cv2.drawContours(img, np.flip(slice_of_interest[:, :2], axis=1).reshape(-1, 1, 2), -1, (255, 0, 0), 2)

        cv2.imshow('TF2 Detection', img)
        cv2.waitKey(200)
        slice_num += 1
    preds.to_pickle("./my_pred.pkl")


def concat_anns(scan_obj):
    anns = scan_obj.annotations
    # If nodule is malign
    anns_contours = [sorted(ann.contours, key=lambda c: c.image_z_position) for ann in anns if ann.calcification > 4]

    slice_contours_matrix = []
    for ann_contours in anns_contours:
        for slice in ann_contours:
            contour_coords = slice.to_matrix()
            slice_contours_matrix.append(contour_coords)
    slice_contours_matrix = np.array(sorted(slice_contours_matrix, key=lambda x: x[0][2]))

    return anns, slice_contours_matrix


def extract_img_from_scan(scan, slice_num):
    img_m1 = scan[:, :, slice_num - 1]
    img_0 = scan[:, :, slice_num]
    img_p1 = scan[:, :, slice_num + 1]
    img = cv2.merge((img_m1, img_0, img_p1))
    img = normalizePlanes(img)
    return img


def detect_from_video(detector, video_path, display_det=False, save_output=False, output_dir='output/'):
    scan = CTScan(video_path)
    scan.load_image()

    if save_output:
        output_path = os.path.join(output_dir, 'detection_' + video_path.split("/")[-1])
        frame_width = int(scan.shape[1])
        frame_height = int(scan.shape[2])
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height))

    slice_num = 1
    preds = {}
    while slice_num < scan.shape[0] - 1:
        #
        # Preprocess data
        #

        # Stack images
        img_m1 = scan.get_slice(slice_num - 1)
        img_0 = scan.get_slice(slice_num)
        img_p1 = scan.get_slice(slice_num + 1)
        img = cv2.merge((img_m1, img_0, img_p1))

        print(f"Processing CT Scan: {video_path.split('/')[-1]} - Slice: {slice_num}")
        timestamp1 = time.time()
        det_boxes = detector.DetectFromImage(img)
        elapsed_time = round((time.time() - timestamp1) * 1000)  # ms

        preds[slice_num] = det_boxes

        if display_det:
            img = detector.DisplayDetections(img, scan.shape[1], det_boxes, det_time=elapsed_time)
            cv2.imshow('TF2 Detection', img)

        if cv2.waitKey(1) == 27: break

        if save_output:
            out.write(img)

        slice_num += 1

    if save_output:
        out.release()

    return preds


def detect_images_from_folder(detector, images_dir, save_output=False, output_dir='output/'):
    for file in os.scandir(images_dir):
        if file.is_file():  # and file.name.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(images_dir, file.name)
            print(image_path)
            img = cv2.imread(image_path)
            det_boxes = detector.DetectFromImage(img)
            img = detector.DisplayDetections(img, det_boxes)

            cv2.imshow('TF2 Detection', img)
            cv2.waitKey(0)

            if save_output:
                img_out = os.path.join(output_dir, file.name)
                cv2.imwrite(img_out, img)


if __name__ == "__main__":
    visualize_patch()
    exit()

    parser = argparse.ArgumentParser(description='Object Detection from Images or Video')
    parser.add_argument('--model_path', help='Path to frozen detection model',
                        default=r'C:\Users\bardh\tensorflow\model_zoo\efficientdet_d0_coco17_tpu-32\v0.full_subset0_1_3D_2mm\exported_models\saved_model')
    parser.add_argument('--path_to_labelmap', help='Path to labelmap (.pbtxt) file',
                        default='../preprocessing_and_detection/processed_data/label_map.pbtxt')
    parser.add_argument('--class_ids', help='id of classes to detect, expects string with ids delimited by ","',
                        type=str, default=None)  # example input "1,3" to detect person and car
    parser.add_argument('--threshold', help='Detection threshold', type=float, default=0.05)
    parser.add_argument('--images_dir', help='Directory to input images)', default='data/samples/images/')
    parser.add_argument('--video_path', help='Path to input video)',
                        default='../data/subset_012345/1.3.6.1.4.1.14519.5.2.1.6279.6001.303494235102183795724852353824.mhd')
    parser.add_argument('--output_directory', help='Path to output images and video', default='data/samples/output')
    parser.add_argument('--display_detections', help='Displays the detections on screen', default=1)
    parser.add_argument('--video_input', help='Flag for video input, default: False', default=True,
                        action='store_true')  # default is false
    parser.add_argument('--save_output',
                        help='Flag for save images and video with detections visualized, default: False',
                        action='store_true')  # default is false
    args = parser.parse_args()

    id_list = None
    if args.class_ids is not None:
        id_list = [int(item) for item in args.class_ids.split(',')]

    if args.save_output:
        if not os.path.exists(args.output_directory):
            os.makedirs(args.output_directory)

    # instance of the class DetectorTF2
    detector = DetectorTF2(args.model_path, args.path_to_labelmap, class_id=id_list, threshold=args.threshold)

    if args.video_input:
        preds = detect_from_video(None, args.video_path, display_det=args.display_detections,
                                  save_output=args.save_output,
                                  output_dir=args.output_directory)

    else:
        detect_images_from_folder(detector, args.images_dir, save_output=args.save_output,
                                  output_dir=args.output_directory)

    print("Done ...")
    cv2.destroyAllWindows()
