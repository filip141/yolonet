import os
import numpy as np
from xml.dom import minidom
from scripts.utils.yoloutils import class2id, DB_FILES, VOC_CLASSES


class Database(object):

    def __init__(self, name):
        self.annotations, self.images = self.load_database(name)

    def load_database(self, name):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        ann_path = os.path.join(curr_dir, "../..", DB_FILES[name]['annotations'])
        im_path = os.path.join(curr_dir, "../..", DB_FILES[name]['images'])
        return self.ann_xml_gen(ann_path), self.image_gen(im_path)

    @staticmethod
    def image_gen(path):
        while True:
            for d_file in os.listdir(path):
                d_file_path = os.path.join(path, d_file)
                # Check if file exist
                file_desc = open(d_file_path)
                yield file_desc

    @staticmethod
    def ann_xml_gen(path):
        while True:
            for d_file in os.listdir(path):
                d_file_path = os.path.join(path, d_file)
                yield minidom.parse(d_file_path)

    def db_iter(self, side=7, boxes=3, classes=20):
        annotation_gen = self.annotations
        img_gen = self.images
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
                    box_w = model_width
                    box_h = model_height
                    confs[int((ind_y * side) + ind_x), it_box] = 1.0
                    cords[int((ind_y * side) + ind_x), it_box] = [box_x, box_y, box_w, box_h]
            yield im_file, np.hstack((probs.flatten(), confs.flatten(), cords.flatten()))


if __name__ == '__main__':
    voc = Database('voc2007')

