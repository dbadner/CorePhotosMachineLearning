import ctypes
import os
import warnings

import Detectron2WBEval as Wb
import FormBrowse as Bws
import FormUI as Ui
import OCR_Root as Ocr
import SkipDetectron as Sd


def main():
    skip_detectron = False  # skip detectron whiteboard recognition, default = false, for development

    warnings.simplefilter(action='ignore', category=FutureWarning)

    obj_bws = Bws.FrmBrowse()
    cpu_mode = obj_bws.cpuMode
    skip_ml = obj_bws.skipML

    # read in inputdirectory from user, and generate nested output directories if they do not yet exist
    inputdir = obj_bws.ImagePath.get()
    output_wb_dir = inputdir + "/" + "Output_WB"
    if not os.path.exists(output_wb_dir):
        os.makedirs(output_wb_dir)
    output_anno_dir = inputdir + "/" + "Output_Anno"
    if not os.path.exists(output_anno_dir):
        os.makedirs(output_anno_dir)
    output_wb_anno_dir = inputdir + "/" + "Output_WB_Anno"
    if not os.path.exists(output_wb_anno_dir):
        os.makedirs(output_wb_anno_dir)
    output_named_dir = inputdir + "/" + "Output_Named_Images"
    if not os.path.exists(output_named_dir):
        os.makedirs(output_named_dir)

    cf_output: list
    if not skip_ml:
        wb_output_list = []
        error_count = 0
        if not skip_detectron:
            print("Reading in images and searching for white boards...")
            obj_wb = Wb.FindWhiteBoards(inputdir, output_wb_dir, output_wb_anno_dir)
            wb_output_list, error_count = obj_wb.run_model(True, True, cpu_mode)
        else:
            # use what is already in the directory, for debugging
            wb_output_list, error_count = Sd.skip_detectron(inputdir, output_wb_dir, output_wb_anno_dir)
        if error_count > 0:
            ctypes.windll.user32.MessageBoxW(0,
                                             "Warning: white board not found in {} image file(s). "
                                             "Refer to output in terminal for details.".format(
                                                 error_count), "Warning", 0)

        print("Classifying text in photos...")
        cf_output = Ocr.ocr_root(wb_output_list, output_wb_dir, output_anno_dir)

        print("Classification complete. Stepping through results interactively...")
    else:
        # skipping machine learning part of program for efficiency, straight to UIForm
        print("Skipping machine learning. Straight to user form to manually rename photos.")
        cf_output = Ocr.read_images_root(inputdir)

    obj_ui = Ui.UIForm(output_wb_dir, output_named_dir, cf_output, skip_ml)


if __name__ == '__main__':
    main()
