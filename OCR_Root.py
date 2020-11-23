import Functions as fn
import cv2
import numpy as np
import OCR_Handwriting as hw
import os


def ocr_root(wb_list, input_dir, output_anno_dir):
    # function instantiates OCR_Handwriting images and returns necessary results for FormUI
    # wb_list is a list of tuple of [image filename, image filepath, whiteboard output image filepath,
    # annotated output image filepath]

    cf_output = []  # list of CfOutputObj class objects containing output from the ocr classification script that is
    # returned from this function and fed
    # into the user form
    # parameters: [image filename, image filepath, whiteboard output image filepath, annotated output image filepath,
    # annotated whiteboard output image filepath, classified depth_from, classified depth_to, classified wet/dry,
    # depth_from probability as %, depth_to probability as %

    for wb in wb_list:  # for image_file in os.listdir(self.InputDir):
        image_file = wb[0]  # retrieve the whiteboard image file name
        image_path = wb[2]  # retrieve the whiteboard image path
        # load the input file from disk
        # image_path = self.InputDir + '/' + image_file
        image = cv2.imread(image_path)
        if type(image) is np.ndarray:  # only process if image file
            # create new image object
            print("Processing characters in image: " + image_file)
            wbimage = hw.WBImage(input_dir)
            wbimage.image = fn.resize_image(image, 2000, 2000)
            wbimage.preprocess()  # preprocess the image, convert to gray
            wbimage.find_chars_words()
            # find characters and words, store as [image, (x, y, w, h)]
            wb_anno_out_path = wbimage.run_model(image_file, output_anno_dir)  # run the model to predict characters,
            # return saved image output path
            cfobj = CfOutputObj(wb[0], wb[1], wb[3], wb_anno_out_path, wbimage.depthFrom, wbimage.depthTo,
                                wbimage.wetDry, wbimage.depthFromP, wbimage.depthToP, wbimage.wetDryP)
            cf_output.append(cfobj)
            # cf_output.append((wb[0], wb[1], wb[2], wb[3], wb_anno_out_path, wbimage.depth_from, wbimage.depth_to,
            # wbimage.wet_dry, wbimage.depth_from_p, wbimage.depth_to_p, wbimage.wet_dry_p))

    return cf_output


def read_images_root(input_dir):
    # This is executed if the user has selected to skip machine learning, just need to read in image files
    cf_output = []
    for image_file in os.listdir(input_dir):
        image_path = input_dir + '/' + image_file
        image: np.ndarray = cv2.imread(image_path)
        if type(image) is np.ndarray:  # only process if image file
            cfobj = CfOutputObj(image_file, image_path, image_path, image_path, "", "", "", -1, -1, -1)
            cf_output.append(cfobj)
    return cf_output


class CfOutputObj:

    def __init__(self, img_file_name, img_file_path, img_wb_anno_file_path, img_anno_file_path, depth_from,
                 depth_to, wet_dry, depth_from_p, depth_to_p, wet_dry_p):
        self.ImgFileName = img_file_name
        self.ImgFilePath = img_file_path
        self.ImgWBAnnoFilePath = img_wb_anno_file_path
        self.ImgAnnoFilePath = img_anno_file_path
        self.DepthFrom = depth_from
        self.DepthTo = depth_to
        self.WetDry = wet_dry
        self.DepthFromP = depth_from_p
        self.DepthToP = depth_to_p
        self.wetDryP = wet_dry_p
