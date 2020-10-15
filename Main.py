import Detectron2WBEval as wb
import OCR_Handwriting as hw
import tesseract_wordboxes as tesswb
from detectron2.structures import Instances
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

inputdir = r'inputimages'
outputdir = r'inputimages\output'


#objWB = wb.FindWhiteBoards(inputdir, outputdir)
#wbextents: dict = objWB.RunModel(True, False)


objHW = hw.FindCharsWords(outputdir)
objHW.OCRHandwriting()

#objTessWB = tesswb.TessFindWords(inputdir)
#objTessWB.RunTess(True, True) #, wbextents)



#temp = wbOutputDict['RC643_008.29-017.56_m_DRY.JPG']

