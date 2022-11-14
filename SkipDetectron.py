import numpy as np
import cv2
import os


def skip_detectron(inputdir, output_wb_dir, output_wb_anno_dir):
    output_list = []
    for image_file in os.listdir(output_wb_dir):
        image_path = output_wb_dir + '/' + image_file
        img: np.ndarray = cv2.imread(image_path)

        if type(img) is np.ndarray:  # only process if image file
            output_list.append((image_file, image_path, image_path, image_path))

    return output_list, 0 #error count = 0
