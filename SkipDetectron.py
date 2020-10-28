import numpy as np
import cv2
import os

def skip_detectron(inputdir, outputWBDir, outputWBAnnoDir):
    outputList = []
    for image_file in os.listdir(outputWBDir):
        image_path = outputWBDir + '\\' + image_file
        img: np.ndarray = cv2.imread(image_path)

        if type(img) is np.ndarray:  # only process if image file
            outputList.append((image_file, image_path, image_path, image_path))
            # wbOutputList [image filename, image filepath, whiteboard output image filepath, annotated output image filepath]

    return outputList, 0 #error count = 0