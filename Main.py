import Detectron2WBEval as wb
import OCR_Handwriting as hw
#import tesseract_wordboxes as tesswb
from detectron2.structures import Instances
import FormBrowse as bws
import FormUI as ui
import warnings
import os
import ctypes

#def DefCreateOutDirs(inputdir):


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

print("Reading in images and searching for white boards...")
objWB = wb.FindWhiteBoards(inputdir, outputWBDir, outputWBAnnoDir)
wbOutputList, errorCount = objWB.RunModel(True, True)
#wbOutputList [image filename, image filepath, whiteboard output image filepath, annotated output image filepath]

if errorCount > 0:
    ctypes.windll.user32.MessageBoxW(0, "Warning: white board not found in {} image file(s). Refer to output in terminal for details.".format(errorCount), "Warning", 0)

print("Stepping through whiteboards interactively and classifying text...")
#objHW = hw.FindCharsWords(outputWBDir, outputNamedDir)
#objHW.OCRHandwriting(wbOutputList)
objUI = ui.UIForm(outputWBDir, outputNamedDir, wbOutputList)
# objUI.OCR(wbOutputList[0]) #call OCR method to read in first whiteboard image

#objTessWB = tesswb.TessFindWords(inputdir)
#objTessWB.RunTess(True, True) #, wbextents)



#temp = wbOutputDict['RC643_008.29-017.56_m_DRY.JPG']

