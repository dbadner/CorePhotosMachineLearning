# Code modified from: https://towardsdatascience.com/understanding-detectron2-demo-bc648ea569e5
# trained model to find whiteboards in photographs using detectron2
import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
import re
from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer, VisImage
import os
from detectron2.data.datasets import register_coco_instances


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

    def register_dataset(self):
        # register the training dataset, only need to do this once
        catalog_list = MetadataCatalog.list()
        if 'wb_train' not in catalog_list:
            register_coco_instances("wb_train", {}, "roboflow/train/_annotations.coco.json", "/roboflow/train")
        if 'wb_val' not in catalog_list:
            register_coco_instances("wb_val", {}, "roboflow/valid/_annotations.coco.json", "/roboflow/valid")
        if 'wb_test' not in catalog_list:
            register_coco_instances("wb_test", {}, "roboflow/test/_annotations.coco.json", "/roboflow/test")

    def run_model(self, save_crop_output: bool, save_anno_output: bool, cpu_mode: bool):
        # set to CPU mode if system does not have NVIDIA GPU
        output_list = []

        self.register_dataset()

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
        cfg.MODEL.WEIGHTS = "wb_model.pth"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        if cpu_mode:
            cfg.MODEL.DEVICE = 'cpu'
        predictor = DefaultPredictor(cfg)
        error_count: int = 0

        for image_file in os.listdir(self.InputDir):
            image_path = self.InputDir + '/' + image_file
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

                anno_out_filename = ""
                if save_anno_output:
                    # draw output and save to png
                    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("wb_test"), scale=1.0)
                    result: VisImage = v.draw_instance_predictions(output.to("cpu"))
                    result_image: np.ndarray = result.get_image()[:, :, ::-1]

                    # get file name without extension, -1 to remove "." at the end
                    anno_out_filename: str = self.OutputWBAnnoDir + '/' + re.search(r"(.*)\.", image_file).group(0)[:-1]
                    anno_out_filename += "_WB_Anno.png"
                    cv2.imwrite(anno_out_filename, result_image)

                if len(scores) > 0:
                    if save_crop_output:
                        # crop and save the image
                        # https://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
                        crop_img = img[box[1].astype(int):box[3].astype(int), box[0].astype(int):box[2].astype(int)]
                        # get file name without extension, -1 to remove "." at the end
                        out_file_name: str = self.OutputWBDir + '/' + re.search(r"(.*)\.", image_file).group(0)[:-1]
                        out_file_name += "_WB_Cropped.png"
                        cv2.imwrite(out_file_name, crop_img)
                        # add to the output list
                        output_list.append((image_file, image_path, out_file_name, anno_out_filename))
                else:
                    print("WARNING: WHITE BOARD NOT FOUND IN IMAGE FILE: " + image_file + ". SKIPPING IMAGE.")
                    error_count += 1

        return output_list, error_count
