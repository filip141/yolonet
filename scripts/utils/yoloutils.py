import os
import numpy as np

DB_FILES = {
    "voc2007": {
        "annotations": "data/VOC2007/Annotations",
        "images": "data/VOC2007/JPEGImages",
    }
}

VOC_CLASSES = [
    "person", "bird", "cat", "cow", "dog", "horse", "sheep",
    "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
    "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"
]


def class2id(class_name):
    return VOC_CLASSES.index(class_name)
