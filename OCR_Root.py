from PIL import Image, ImageTk
import Functions as fn
import cv2
import numpy as np
import OCR_Handwriting as hw
import os
import shutil
import ctypes
import re


def ocrRoot(wbList, inputDir, outputAnnoDir):
    #function instantiates OCR_Handwriting images and returns necessary results for FormUI
    # wbList is a list of tuple of [image filename, image filepath, whiteboard output image filepath, annotated output image filepath]

    cfOutput = [] #list of cfOutputObj class objects containing output from the OCR classification script that is returned from this function and fed
    #into the user form
    #parameters: [image filename, image filepath, whiteboard output image filepath, annotated output image filepath,
    #annotated whiteboard output image filepath, classified depthFrom, classified depthTo, classified wet/dry,
    #depthFrom probability as %, depthTo probability as %

    for wb in wbList:  # for image_file in os.listdir(self.InputDir):
        image_file = wb[0]  # retrieve the whiteboard image file name
        image_path = wb[2]  # retrieve the whiteboard image path
        # load the input file from disk
        # image_path = self.InputDir + '/' + image_file
        image = cv2.imread(image_path)
        if type(image) is np.ndarray:  # only process if image file
            # create new image object
            print("Processing characters in image: " + image_file)
            wbimage = hw.WBImage(inputDir)
            wbimage.image = fn.ResizeImage(image, 2000, 2000)
            wbimage.Preprocess()  # preprocess the image, convert to gray
            wbimage.FindCharsWords()
            # find characters and words, store as [image, (x, y, w, h)]
            wbAnnoOutPath = wbimage.RunModel(image_file, outputAnnoDir)  # run the model to predict characters, return saved image output path
            cfobj = cfOutputObj(wb[0], wb[1], wb[3], wbAnnoOutPath, wbimage.depthFrom, wbimage.depthTo, wbimage.wetDry,
                             wbimage.depthFromP, wbimage.depthToP, wbimage.wetDryP)
            cfOutput.append(cfobj)
            #cfOutput.append((wb[0], wb[1], wb[2], wb[3], wbAnnoOutPath, wbimage.depthFrom, wbimage.depthTo, wbimage.wetDry,
            #                 wbimage.depthFromP, wbimage.depthToP, wbimage.wetDryP))

    return cfOutput

def readImagesRoot(inputDir):
    #This is executed if the user has selected to skip machine learning, just need to read in image files
    cfOutput = []
    for image_file in os.listdir(inputDir):
        image_path = inputDir + '/' + image_file
        image: np.ndarray = cv2.imread(image_path)
        if type(image) is np.ndarray:  # only process if image file
            cfobj = cfOutputObj(image_file, image_path, image_path, image_path, "", "", "", -1, -1, -1)
            cfOutput.append(cfobj)
    return cfOutput

class cfOutputObj:

    def __init__(self, imgFileName, imgFilePath, imgWBAnnoFilePath, imgAnnoFilePath, depthFrom, depthTo, wetDry, depthFromP, depthToP, wetDryP):
        self.ImgFileName = imgFileName
        self.ImgFilePath = imgFilePath
        self.ImgWBAnnoFilePath = imgWBAnnoFilePath
        self.ImgAnnoFilePath = imgAnnoFilePath
        self.DepthFrom = depthFrom
        self.DepthTo = depthTo
        self.WetDry = wetDry
        self.DepthFromP = depthFromP
        self.DepthToP = depthToP
        self.wetDryP = wetDryP
