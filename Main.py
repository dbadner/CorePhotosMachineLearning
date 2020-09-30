import Detectron2WBEval as wb
from detectron2.structures import Instances

obj = wb.FindWhiteBoards(r'C:\Users\DBadner\Desktop\input')
wbOutputDict: dict = obj.RunModel(True)

#temp = wbOutputDict['RC643_008.29-017.56_m_DRY.JPG']

