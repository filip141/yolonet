import os
from xml.dom import minidom
import cv2

import numpy as np

DB_FILES = {
    "voc2007": {
        "annotations": "data/VOC2007/Annotations",
        "images": "data/VOC2007/JPEGImages",
    },
    "voc2012": {
        "annotations": "data/VOC2012/Annotations",
        "images": "data/VOC2012/JPEGImages",
    }
}

VOC_CLASSES = [
    "person", "bird", "cat", "cow", "dog", "horse", "sheep",
    "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
    "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"
]


def path_sort(record):
    return "{}{}".format(record[-10:-4], record[-15:-11])


def class2id(class_name):
    return VOC_CLASSES.index(class_name)


class Database(object):
    def __init__(self, name='voc2007', validation_split=0.1):
        self.test_annotations, self.test_images, self.validation_annotations, self.validation_images, self.set_size = \
            self.load_database(name, validation_split)

    def load_database(self, name, validation_split):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        ann_path = os.path.join(curr_dir, "..", DB_FILES[name]['annotations'])
        im_path = os.path.join(curr_dir, "..", DB_FILES[name]['images'])
        ann_paths_list = [os.path.join(ann_path, x) for x in os.listdir(ann_path)]
        im_patchs_list = [os.path.join(im_path, x) for x in os.listdir(im_path)]
        ann_paths_list = sorted(ann_paths_list, key=path_sort)
        im_patchs_list = sorted(im_patchs_list, key=path_sort)

        split_boundary = int((1 - validation_split) * len(im_patchs_list))
        set_size = len(im_patchs_list)
        return self.ann_xml_gen(ann_paths_list[:split_boundary]), self.image_gen(im_patchs_list[:split_boundary]), \
               self.ann_xml_gen(ann_paths_list[split_boundary:]), self.image_gen(im_patchs_list[split_boundary:]), \
               set_size

    @staticmethod
    def image_gen(d_files_paths):
        while True:
            for d_file_path in d_files_paths:
                # Check if file exist
                file_desc = cv2.imread(d_file_path, 1)
                file_desc = cv2.cvtColor(file_desc, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(file_desc, (224, 224))
                image_data = np.array(img_resized, dtype='float32')
                image_data /= 255.
                image_data = np.transpose(image_data, (2, 0, 1))
                image_data = np.expand_dims(image_data, 0)
                yield image_data

    @staticmethod
    def ann_xml_gen(d_files_paths):
        while True:
            for d_file_path in d_files_paths:
                yield minidom.parse(d_file_path)

    def sb_batch_iter(self, batch_size=32, side=7, boxes=3, classes=20, type='train'):
        assert type in ['test', 'train', 'val', 'validation']
        if type in ['val', 'validation']:
            db_iter = self.db_iter(self.validation_annotations, self.validation_images, side=side, boxes=boxes,
                                   classes=classes)
        elif type == 'train':
            db_iter = self.db_iter(self.test_annotations, self.test_images, side=side, boxes=boxes, classes=classes)
        while True:
            batch_x = np.zeros((batch_size, 3, 224, 224))
            batch_y = np.zeros((batch_size, 1470))
            for idx in range(batch_size):
                x, y = next(db_iter, None)
                batch_x[idx] = x[0]
                batch_y[idx] = y
            yield batch_x, batch_y

    def db_iter(self, annotation_gen, img_gen, side=7, boxes=3, classes=20):
        while True:
            ann_file = next(annotation_gen, None)
            im_file = next(img_gen, None)
            # Get image size
            size_section = ann_file.getElementsByTagName('size')
            im_width = float(size_section[0].getElementsByTagName('width')[0].firstChild.nodeValue)
            im_height = float(size_section[0].getElementsByTagName('height')[0].firstChild.nodeValue)

            # Define matrixes
            SS = side * side
            probs = np.zeros([SS, classes])
            confs = np.zeros([SS, boxes])
            cords = np.zeros([SS, boxes, 4])
            # Iterate over objects in xml file
            object_list = ann_file.getElementsByTagName('object')
            for voc_object in object_list:
                # Get class name and class id
                class_name = voc_object.getElementsByTagName('name')[0].firstChild.nodeValue
                class_id = class2id(class_name)
                # Get box coords
                bndbox = voc_object.getElementsByTagName('bndbox')
                xmin = float(bndbox[0].getElementsByTagName('xmin')[0].firstChild.nodeValue)
                ymin = float(bndbox[0].getElementsByTagName('ymin')[0].firstChild.nodeValue)
                xmax = float(bndbox[0].getElementsByTagName('xmax')[0].firstChild.nodeValue)
                ymax = float(bndbox[0].getElementsByTagName('ymax')[0].firstChild.nodeValue)
                model_width = xmax - xmin
                model_height = ymax - ymin
                model_x = (xmax + xmin) / 2.0
                model_y = (ymax + ymin) / 2.0

                # Set box parameters
                grid_x_width = im_width / float(side)
                grid_y_height = im_height / float(side)
                ind_x = int(model_x / grid_x_width)
                ind_y = int(model_y / grid_y_height)
                # Set class id
                probs[int((ind_y * side) + ind_x), class_id] = 1.0
                for it_box in range(boxes):
                    box_x = (model_x % grid_x_width) / grid_x_width
                    box_y = (model_y % grid_y_height) / grid_y_height
                    box_w = model_width / float(im_width)
                    box_h = model_height / float(im_height)
                    confs[int((ind_y * side) + ind_x), it_box] = 1.0
                    cords[int((ind_y * side) + ind_x), it_box] = [box_x, box_y, box_w, box_h]
            yield im_file, np.hstack((probs.flatten(), confs.flatten(), cords.flatten()))
