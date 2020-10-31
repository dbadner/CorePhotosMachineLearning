import Detectron2WBEval as wb
import SkipDetectron as sd
import FormBrowse as bws
import FormUI as ui
import OCR_Root as ocr
import warnings
import os
import ctypes


def main():
    skipDetectron = False #skip detectron whiteboard recognition, default = false, for development


    warnings.simplefilter(action='ignore', category=FutureWarning)

    objBWS = bws.frmBrowse()

    #read in inputdirectory from user, and generate nested output directories if they do not yet exist
    #inputdir = #r'inputimages'
    inputdir = objBWS.ImagePath.get()
    outputWBDir = inputdir + "\\" + "Output_WB"
    if not os.path.exists(outputWBDir): os.makedirs(outputWBDir)
    outputAnnoDir = inputdir + "\\" + "Output_Anno"
    if not os.path.exists(outputAnnoDir): os.makedirs(outputAnnoDir)
    outputWBAnnoDir = inputdir + "\\" + "Output_WB_Anno"
    if not os.path.exists(outputWBAnnoDir): os.makedirs(outputWBAnnoDir)
    outputNamedDir = inputdir + "\\" + "Output_Named_Images"
    if not os.path.exists(outputNamedDir): os.makedirs(outputNamedDir)


    wbOutputList = []
    errorCount = 0
    if not skipDetectron:
        print("Reading in images and searching for white boards...")
        objWB = wb.FindWhiteBoards(inputdir, outputWBDir, outputWBAnnoDir)
        wbOutputList, errorCount = objWB.RunModel(True, True)
        #wbOutputList [image filename, image filepath, whiteboard output image filepath, annotated output image filepath]

    else:
        #use what is already in the directory, for debugging
        wbOutputList, errorCount = sd.skip_detectron(inputdir, outputWBDir, outputWBAnnoDir)

    if errorCount > 0:
        ctypes.windll.user32.MessageBoxW(0, "Warning: white board not found in {} image file(s). Refer to output in terminal for details.".format(errorCount), "Warning", 0)

    print("Classifying text in photos...")
    cfOutput = ocr.ocrRoot(wbOutputList, outputWBDir, outputAnnoDir)

    print("Classification complete. Stepping through results interactively...")
    objUI = ui.UIForm(outputWBDir, outputNamedDir, cfOutput)


    #objTessWB = tesswb.TessFindWords(inputdir)
    #objTessWB.RunTess(True, True) #, wbextents)

if __name__ == '__main__':
    main()




