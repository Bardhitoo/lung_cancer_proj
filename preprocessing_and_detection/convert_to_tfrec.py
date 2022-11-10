import os

import pandas as pd
import tensorflow as tf

from preprocessing_and_detection.utils.my_utils import *


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


def write_tf_record(imgs_path, X, y, dataset_name):
    """Generates TFRecord file from given list of annotations."""

    out = f"./processed_data/{dataset_name}_{EXPERIMENT_NAME}.tfrecord"
    width = 40
    label_names = {0: [1, "Benign"], 1: [2, "Malignant"]}
    image_dimension = 512
    seen = {}

    with tf.io.TFRecordWriter(out) as writer:
        for index, row in X.iterrows():
            print(f"Processing: {dataset_name} - {index} / {len(X) - 1}")
            sample_name = row['seriesuid']

            if sample_name in seen:
                seen[sample_name] = seen[sample_name] + 1
            else:
                seen[sample_name] = 0
            image_path = os.path.join(imgs_path, dataset_name, sample_name + f"_{seen[sample_name]}" + ".jpg")

            with tf.io.gfile.GFile(image_path, 'rb') as f:
                image_buffer = f.read()

            # image_buffer.size()  # TODO: Figure out if this actually returns the size
            features = {'image/encoded': _bytes_feature(image_buffer),
                        'image/filename': _bytes_feature(f"{sample_name}_{seen[sample_name]}".encode('utf8')),
                        'image/format': _bytes_feature("jpg".encode(encoding='utf-8')),
                        'image/height': _int64_feature(image_dimension),
                        'image/width': _int64_feature(image_dimension)}

            x_center, y_center = row["coordX"], row["coordY"]  # TODO: Is this actually returning the center?
            xmin, ymin = x_center - (width / 2), y_center - (width / 2)
            xmax, ymax = x_center + (width / 2), y_center + (width / 2)

            # bounding box features
            features['image/object/bbox/xmin'] = _float_feature(xmin / image_dimension)
            features['image/object/bbox/ymin'] = _float_feature(ymin / image_dimension)
            features['image/object/bbox/xmax'] = _float_feature(xmax / image_dimension)
            features['image/object/bbox/ymax'] = _float_feature(ymax / image_dimension)

            # label from [0, 2) iterval
            features['image/object/class/label'] = _int64_feature(label_names[int(y.iloc[index])][0])
            features['image/object/class/text'] = _bytes_feature(label_names[int(y.iloc[index])][1].encode('utf8'))

            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())


def main():
    for annotations_name in ["train", "test"]:
        dataset = pd.read_csv(f"./processed_data/{annotations_name}_{EXPERIMENT_NAME}_transformed_coords.csv")
        X = dataset.loc[:, dataset.columns != "class"]
        y = dataset["class"]

        write_tf_record(SAVE_IMG, X, y, annotations_name)


if __name__ == "__main__":
    main()
