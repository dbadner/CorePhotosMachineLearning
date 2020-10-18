# Code modified from: https://towardsdatascience.com/understanding-detectron2-demo-bc648ea569e5
#trained model to find whiteboards in photographs using detectron2

import cv2
import numpy as np
import re
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer, VisImage
import os
from detectron2.data.datasets import register_coco_instances
import tensorflow as tf
from PIL import Image
import torch


class FindWhiteBoards:
    # class variables
    InputDir: str  # input directory path with photographs
    OutputWBDir: str  # output directory for whiteboard images
    OutputWBAnnoDir: str # output directory for annotated input image showing whiteboard location

    # parameterized constructor
    def __init__(self, inputdir: str, outputWBDir: str, outputWBAnnoDir: str):
        self.InputDir = inputdir
        self.OutputWBDir = outputWBDir
        self.OutputWBAnnoDir = outputWBAnnoDir

    def RegisterDataset(self):
        # register the training dataset, only need to do this once
        register_coco_instances("wb_train", {}, "roboflow/train/_annotations.coco.json", "/roboflow/train")
        register_coco_instances("wb_val", {}, "roboflow/valid/_annotations.coco.json", "/roboflow/valid")
        register_coco_instances("wb_test", {}, "roboflow/test/_annotations.coco.json", "/roboflow/test")

    def RunModel(self, saveCropOutput: bool, saveAnnoOutput: bool):
        #outputDict = {}
        outputList = []

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
        cfg.MODEL.WEIGHTS = "./wb_model/model_final.pth"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        predictor = DefaultPredictor(cfg)
        errorCount: int = 0

        # image_file = "input/RC663_0040.76-0047.60_m_DRY.jpg"

        for image_file in os.listdir(self.InputDir):
            image_path = self.InputDir + '\\' + image_file
            img: np.ndarray = cv2.imread(image_path)

            if type(img) is np.ndarray:  # only process if image file
                print("Processing image: " + image_file)
                output: Instances = predictor(img)["instances"]  # predict

                obj: dict = output.get_fields()

                scores: np.ndarray = obj['scores'].cpu().numpy()
                maxscore: float = 0
                indmaxscore: int = 0
                for i in range(len(scores) - 1):
                    if scores[i] > maxscore:
                        maxscore = scores[i]
                        indmaxscore = i

                if len(scores) > 0:
                    box: np.ndarray = obj['pred_boxes'].tensor.cpu().numpy()[indmaxscore]
                else:
                    box = np.ones(1) * (-1)

                # outputlist.append(output)
                #outputDict[image_file] = box


                anno_out_filename = ""
                if saveAnnoOutput:
                    # draw output and save to png
                    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("wb_test"), scale=1.0)
                    result: VisImage = v.draw_instance_predictions(output.to("cpu"))
                    result_image: np.ndarray = result.get_image()[:, :, ::-1]

                    # get file name without extension, -1 to remove "." at the end
                    anno_out_filename: str = self.OutputWBAnnoDir + '\\' + re.search(r"(.*)\.", image_file).group(0)[:-1]
                    anno_out_filename += "_WB_Anno.png"
                    cv2.imwrite(anno_out_filename, result_image)

                    # code for displaying image:
                    # imgout = cv2.imread(out_file_name)
                    # cv2.imshow('Output Image', imgout)

                    # cv2.waitKey(0)
                if len(scores) > 0:
                    if saveCropOutput:
                        # crop and save the image
                        # https://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
                        crop_img = img[box[1].astype(int):box[3].astype(int), box[0].astype(int):box[2].astype(int)]
                        # get file name without extension, -1 to remove "." at the end
                        out_file_name: str = self.OutputWBDir + '\\' + re.search(r"(.*)\.", image_file).group(0)[:-1]
                        out_file_name += "_WB_Cropped.png"
                        cv2.imwrite(out_file_name, crop_img)
                        #add to the outputDictionary
                        #outputDict[image_file] = (out_file_name, anno_out_filename)
                        outputList.append((image_file, image_path, out_file_name, anno_out_filename))
                else:
                    print("WARNING: WHITE BOARD NOT FOUND IN IMAGE FILE: " + image_file + ". SKIPPING IMAGE.")
                    errorCount += 1

        return outputList, errorCount
