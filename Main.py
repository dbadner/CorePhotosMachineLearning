import Detectron2WBEval as wb
import tesseract_wordboxes as tesswb
from detectron2.structures import Instances

inputdir = r'C:\Users\DBadner\Desktop\input'
outputdir = r'C:\Users\DBadner\Desktop\input\output'

#objWB = wb.FindWhiteBoards(inputdir, outputdir)
#wbextents: dict = objWB.RunModel(True, True)

objTessWB = tesswb.TessFindWords(inputdir)
objTessWB.RunTess(True, True) #, wbextents)



#temp = wbOutputDict['RC643_008.29-017.56_m_DRY.JPG']

