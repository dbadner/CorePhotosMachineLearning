# Code modified from: https://towardsdatascience.com/understanding-detectron2-demo-bc648ea569e5

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
    #class variables
    InputDir: str  #input directory path with photographs
    OutputDir: str #output directory for temp output files
    
    # parameterized constructor
    def __init__(self, inputdir: str, outputdir: str):
        self.InputDir = inputdir
        self.OutputDir = outputdir

    def RegisterDataset(self):
        # register the training dataset, only need to do this once
        register_coco_instances("wb_train", {}, "roboflow/train/_annotations.coco.json", "/roboflow/train")
        register_coco_instances("wb_val", {}, "roboflow/valid/_annotations.coco.json", "/roboflow/valid")
        register_coco_instances("wb_test", {}, "roboflow/test/_annotations.coco.json", "/roboflow/test")

    def RunModel(self, saveCropOutput: bool, saveAnnoOutput: bool):
        outputDict = {}

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
        cfg.MODEL.WEIGHTS = "./wb_model/model_final.pth"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        predictor = DefaultPredictor(cfg)

        # image_file = "input/RC663_0040.76-0047.60_m_DRY.jpg"

        for image_file in os.listdir(self.InputDir):
            image_path = self.InputDir + '\\' + image_file
            img: np.ndarray = cv2.imread(image_path)

            if type(img) is np.ndarray: #only process if image file

                output: Instances = predictor(img)["instances"] #predict

                obj: dict = output.get_fields()

                scores: np.ndarray = obj['scores'].cpu().numpy()
                maxscore: float = 0
                indmaxscore: int = 0
                for i in range(len(scores)-1):
                    if scores[i] > maxscore:
                        maxscore = scores[i]
                        indmaxscore = i

                if len(scores) > 0:
                    box: np.ndarray = obj['pred_boxes'].tensor.cpu().numpy()[indmaxscore]
                else:
                    box = np.ones(1)*(-1)

                # outputlist.append(output)
                outputDict[image_file] = box

                if saveCropOutput and len(scores) > 0:
                    #crop and save the image
                    #https://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
                    crop_img = img[box[1].astype(int):box[3].astype(int), box[0].astype(int):box[2].astype(int)]
                    # get file name without extension, -1 to remove "." at the end
                    out_file_name: str = self.OutputDir + '\\' + re.search(r"(.*)\.", image_file).group(0)[:-1]
                    out_file_name += "_cropped.png"
                    cv2.imwrite(out_file_name, crop_img)

                if saveAnnoOutput:
                    #draw output and save to png
                    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("wb_test"), scale=1.0)
                    result: VisImage = v.draw_instance_predictions(output.to("cpu"))
                    result_image: np.ndarray = result.get_image()[:, :, ::-1]

                    # get file name without extension, -1 to remove "." at the end
                    out_file_name: str = self.OutputDir + '\\' + re.search(r"(.*)\.", image_file).group(0)[:-1]
                    out_file_name += "_processed.png"

                    cv2.imwrite(out_file_name, result_image)

                    #code for displaying image:
                    # imgout = cv2.imread(out_file_name)
                    # cv2.imshow('Output Image', imgout)

                    # cv2.waitKey(0)

        return outputDict