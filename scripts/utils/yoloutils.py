import os
import numpy as np

DB_FILES = {
    "voc2007": {
        "annotations": "data/VOC2007/Annotations",
        "images": "data/VOC2007/JPEGImages",
    }
}


class Box(object):
    def __init__(self, width, height, conf, prob):
        self.conf = conf
        self.width, self.height = width, height
        self.prob = prob


class OutputGrid(object):
    def __init__(self, side=7, boxes=2, classes=20):
        self.grid = [[[[Box(0, 0, 0, 0) for b in range(boxes)],
                       [0.0 for c in range(classes)]] for s in range(side)] for s in range(side)]


def data_gen(path):
    for d_file in os.listdir(path):
        d_file_path = os.path.join(path, d_file)
        # Check if file exist
        file_desc = open(d_file_path)
        yield file_desc


def load_database(name):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ann_path = os.path.join(curr_dir, "../..", DB_FILES[name]['annotations'])
    im_path = os.path.join(curr_dir, "../..", DB_FILES[name]['images'])
    return data_gen(ann_path), data_gen(im_path)

if __name__ == '__main__':
    load_database('voc2007')
    og = OutputGrid()