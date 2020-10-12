# modified from: https://www.pyimagesearch.com/2020/08/24/ocr-handwriting-recognition-with-opencv-keras-and-tensorflow/
# identify characters within an image
# import the necessary packages
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
import os

"""
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained handwriting recognition model")
args = vars(ap.parse_args())

# load the handwriting OCR model
print("[INFO] loading handwriting OCR model...")
model = load_model(args["model"])
"""

class FindCharsWords:

	InputDir: str #= r'C:\Users\DBadner\Desktop\input'
	Model: object #= load_model('number_az_model_firstpass.h5')

	def __init__(self, inputdir: str):
		self.InputDir = inputdir
		self.Model = load_model('number_az_model.h5')

	def ResizeImage(self, img, maxW, maxH):
		#resizes the image based on the maximum width and maximum height, returns the resized image
		if img.ndim == 2: #black and white
			(tH, tW) = img.shape
		else: #colour
			(tH, tW, tmp) = img.shape
		# if the width is greater than the height (usually will be), resize along the width dimension
		if tW/tH > maxW/maxH: #width is the defining dimension along which to resize
			hCheck = tH/(tW/maxW)
			if hCheck > 1.00001: #check not < 1, will throw error
				img = imutils.resize(img, width=maxW)
			else:
				img = imutils.resize(img, width=maxW, height=1)
		# otherwise, resize along the height
		else:
			wCheck = tW/(tH/maxH)
			if wCheck > 1.00001:#check not < 1, will throw error
				img = imutils.resize(img, height=maxH)
			else:
				img = imutils.resize(img, height=maxH, width=1)
		return img

	def Preprocess(self, image):
		# convert image to grayscale, and blur it to reduce noise
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		#gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)[1]
		# gray = cv2.GaussianBlur(gray, (5, 5), 0)
		# Applied dilation
		kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
		gray = cv2.morphologyEx(gray, cv2.MORPH_ERODE, kernel3)
		#grayS = self.ResizeImage(gray, 800, 800)
		#cv2.imshow("Preprocessed", grayS)
		#cv2.waitKey(0)
		return gray

	def FindCharsWords(self, gray):
		# perform edge detection, find contours in the edge map, and sort the
		# resulting contours from left-to-right, find words
		#fac = 2 #factor by which to temporarily upscale images to improve edge detection
		#tH, tW = gray.shape
		#imgTmp = self.ResizeImage(gray, tW * fac, tH * fac) #temporarily upsize by fac to make edge detection work better

		edged = cv2.Canny(gray, 30, 150)
		#cv2.imshow("padded", edged)
		#cv2.waitKey(0)
		cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #cv2.CHAIN_APPROX_NONE)
		cnts = imutils.grab_contours(cnts)
		cnts = sort_contours(cnts, method="left-to-right")[0]
		# initialize the list of contour bounding boxes and associated
		#cnts = cnts / fac
		# characters that we'll be OCR'ing
		# chars = []  # pairs of character images and dimensions
		# loop over the contours and populate if they pass the criteria
		chars = self.ProcessChars(gray, cnts)
		return chars

	def ProcessChars(self, gray, cnt):
		# function takes in image and contour and filters the characters to those within words,
		# and those with appropriate sizes, and adjusts white space
		chars = []  # pairs of character images and dimensions
		charDict = {} #charID, ndarray[28x28x1] image, (x,y,w,h) of original image
		for i in range(len(cnt)):
			c = cnt[i]
			# compute the bounding box of the contour
			(x, y, w, h) = cv2.boundingRect(c)
			# filter out bounding boxes, ensuring they are neither too small
			# nor too large
			#if (w >= 5 and w <= 150) and (h >= 8 and h <= 120):
			if (w >= 5 and w <= 375) and (h >= 15 and h <= 300):

				# extract the character and threshold it to make the character
				# appear as *white* (foreground) on a *black* background, then
				# grab the width and height of the thresholded image
				roi = gray[y:y + h, x:x + w]
				thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

				thresh = self.ResizeImage(thresh, 22, 22) #resize the image

				# re-grab the image dimensions (now that its been resized)
				# and then determine how much we need to pad the width and
				# height such that our image will be 28x28
				(tH, tW) = thresh.shape
				dX = int(max(6, 28 - tW) / 2.0)
				dY = int(max(6, 28 - tH) / 2.0)
				# pad the image and force 28x28 dimensions
				padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
				padded = cv2.resize(padded, (28, 28))
				#cv2.imshow("padded", padded)
				#cv2.waitKey(0)
				#print("{}, {}".format(w, h))
				# prepare the padded image for classification via our
				# handwriting OCR model
				padded = padded.astype("float32") / 255.0
				padded = np.expand_dims(padded, axis=-1)
				chars.append((padded, (x, y, w, h)))  # update our list of characters that will be OCR'd
				#charDict[i] = (padded, (x, y, w, h))

		# check each character to make sure not overlapping with an another character, discard if so
		#removeList = []
		removeList = self.CheckOverlap(chars)
		for i in range(len(removeList)-1,-1,-1):
			del chars[removeList[i]]

		wordList = []
		#now loop through chars and assign to words
		for i in range(len(chars)):
			char = chars[i]
			(x, y, w, h) = char[1] #read in character dimensions
			#charDict[i] = char #store in dictionary
			fndWord = False
			for j in range(len(wordList)): #loop through existing word list
				word = wordList[j]
				(xW, yW, wW, hW) = word.dims #read in word dimensions
				(xC, yC, wC, hC) = chars[word.charList[len(word.charList)-1]][1] #read in last character dimensions in word
				#compare to determine whether character is part of current word
				xDif = x - (xW+wW) #check x
				if xDif < hW/1.3: #compare to word height to check if close enough to word to be included (i.e. whitespace between)
					#check y-overlap against previous character in word (instead of fill word
					if not(y > (yC + hC) or (y + h) < yC):#need to also check amount of overlap
						ovl = (min(y+h,yC+hC) - max(y,yC)) / (max(y+h,yC+hC) - min(y,yC)) #percentage overlap
						if ovl > 0.3: #set 30% overlap threshold
							htRatio = h / hW
							if htRatio < 2.5 and htRatio > 0.4: #thresholds for height ratios
								if len(word.charList) <= 3 or (len(word.charList) > 2 and (max(xDif,0) - word.avgCharSpac) / hW < .4):
									#final check - look for a change in average spacing between characters in a word
									fndWord = True
									#update wordList parameters
									yWN = min(y,yW)
									hWN = max(y+h, yW+hW) - yWN
									wWN = x + w - xW
									word.charList.append(i)
									word.dims = (xW, yWN, wWN, hWN)
									#assign avg word spacing
									word.avgCharSpac = (word.avgCharSpac + max(xDif,0)) / (len(word.charList) - 1)
									#wordList[j] = word
									#word[0] = (xW, yWN, wWN, hWN)
									#wordList[j] = (((xW, yWN, wWN, hWN), word[1]))
									break  # exit for loop
								#else:
									#xxx = 1
			if not(fndWord): #start a new word
				newWord = Word()
				newWord.dims = char[1]
				newWord.charList = [i]
				wordList.append(newWord)
				#wordList.append((char[1],[i])) #add dimensions of first character and charID to the wordlist
		# final loop to throw out words that only have one character
		for i in range(len(wordList)-1,-1,-1):
			words = wordList[i]
			if len(words.charList) < 2:
				del wordList[i]

		return chars, wordList

	def CheckOverlap(self, chars):
		# check each character to make sure not overlapping with an another character
		#chars [ndarray[28x28x1], (x,y,w,h)] - list of characters
		removeList = []
		for i in range(len(chars)):
			charI = chars[i]
			(x, y, w, h) = charI[1]  # read in character dimensions
			discard: bool = False
			for j in range(len(chars)):
				if j == i:
					continue
				charJ = chars[j]
				(xC, yC, wC, hC) = charJ[1]  # read in character dimensions
				#xFuz = wC * 0.1 #if over 60% overlap in x and y then remove
				#yFuz = hC * 0.1
				if w * h <= wC * hC:  # only discard the smaller of the two
					if ((x > xC and x < xC + wC) or (x + w > xC and x + w < xC + wC)) and ((y > yC and y < yC + hC) or (y + h > yC and y + h < yC + hC)):
					#if x > xC - xFuz and x + w < xC + wC + xFuz and y > yC - yFuz and y + h < yC + hC + yFuz:
						#there is overlap, determine how much as proportion of smaller item
						aOvl = (min(x+w,xC+wC) - max(x,xC)) * (min(y+h,yC+hC) - max(y,yC))
						percOvl = aOvl/(w*h)
						if percOvl > 0.6: # overlap > 60%
							removeList.append(i)
							break
		return removeList

	def RunModel(self, chars, wordList, gray, image):
		#run the model to predict characters

		# function level variables
		# define the list of label names
		imgAnno = image.copy() #make a local copy of coloured image
		labelNames = "0123456789"
		labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		labelNames = [l for l in labelNames]
		probNumList = [] #list corresponding to wordList indices with probability of number
		wordCharList = [] #list of character arrays in words corresponding to wordList
		keyWordList = [] #list of char lists defining keywords
		keyWordList.append(['F','R','O','M'])
		keyWordList.append(['T','O'])
		keyWordList.append(['D','R','Y'])
		keyWordList.append(['W','E','T'])
		maxProbKeyWordList = [] #maximum probability of current word
		maxProbKeyWordListInd = [] #corresponding indices
		keyWordListInd = []
		#maxProbKeyWordList = np.empty(len(keyWordList), dtype=float)  # list of length len(keyWordList) x 2 containing ID of max probability word for each keyword, and corresponding probability
		#keyWordListInd = np.empty(len(keyWordList), dtype=list) #corresponding indices for keyWordList, values from 0 to 35
		for keyWord in keyWordList:
			keyWordInd = []
			for char in keyWord:
				for i in range(len(labelNames)):
					l = labelNames[i]
					if l == char:
						keyWordInd.append(i)
						break
			keyWordListInd.append(keyWordInd)
			maxProbKeyWordList.append(0) #initialize list of max probabilities to 0
			maxProbKeyWordListInd.append(-1) #initialize index to -1

		# extract the bounding box locations and padded characters
		boxes = [b[1] for b in chars]
		chars = np.array([c[0] for c in chars], dtype="float32")
		# OCR the characters using our handwriting recognition model
		preds = self.Model.predict(chars)


		# loop over the predictions and bounding box locations together
		for (pred, (x, y, w, h)) in zip(preds, boxes):
			# for (x, y, w, h) in boxes:
			# find the index of the label with the largest corresponding
			# probability, then extract the probability and label
			i = np.argmax(pred)
			prob = pred[i]
			label = labelNames[i]
			# draw the prediction on the image
			#print("[INFO] {} - {:.2f}%".format(label, prob * 100))
			cv2.rectangle(imgAnno, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.putText(imgAnno, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
		# loop over words
		for wInd in range(len(wordList)):
			words = wordList[wInd]
			#if len(words[1]) > 1: #show wordBoxes if > 1 character
			(x, y, w, h) = words.dims
			cv2.rectangle(imgAnno, (x, y), (x + w, y + h), (255, 51, 204), 2) #rectangle around word

		# show the image
		imageS = self.ResizeImage(imgAnno, 800, 800)
		cv2.imshow("Image", imageS)
		#grayS = self.ResizeImage(gray, 800, 800)
		#cv2.imshow("Gray", grayS)
		cv2.waitKey(0)

		#loop over words again
		for wInd in range(len(wordList)):
			words = wordList[wInd]
			probNum = 0 #probability that word contains a number
			wordChars = []  # list of characters in word
			#loop over characters in word and determine probability of #,
			for i in words.charList: #for each ith character
				prob = preds[i]
				for j in range(10): #loop through probability of number characters
					probNum += prob[j]
				#probNum /= 10
				wordChars.append(labelNames[np.argmax(preds[i])]) #store the max likelihood character in wordCharList
				# loop through keywords


			probNum /= len(words.charList)
			probNumList.append(probNum)
			wordCharList.append(wordChars)
			#else: #length of word = 1
				#probNumList.append(-1) #null val

			#check probability of being a keyword
			for k in range(len(keyWordListInd)):
				keyWordInd = keyWordListInd[k]
				probKeyWord = 0
				if len(words.charList) == len(keyWordInd):  # same length, so check probability
					for i in range(len(keyWordInd)):  # for each ith character
						ind = words.charList[i]
						probKeyWord += preds[ind][keyWordInd[i]] #find probability that ith characters match
					probKeyWord /= len(keyWordInd)
					#now compare to list, and replace if the new highest likelihool of word is found
					if probKeyWord > maxProbKeyWordList[k]: #new highest likelihood found
						maxProbKeyWordList[k] = probKeyWord
						maxProbKeyWordListInd[k] = wInd  # corresponding word index
		#show image of keyWords picked out, and print probabilities correct
		imgKeyWords = image.copy()
		for k in range(len(maxProbKeyWordListInd)):
			temp: str = ""
			for c in keyWordList[k]:
				temp = temp + c
			temp = temp + ": P={:.1f}%".format(maxProbKeyWordList[k]*100)
			print(temp)
			execute = True
			if maxProbKeyWordList[k] < 0.4: #keyword not found with sufficient probability (40%)
				execute = False
			if k == 2 and maxProbKeyWordList[2] < maxProbKeyWordList[3]:#compare probability of dry vs wet, only show the higher probability
				execute = False
			elif k == 3 and maxProbKeyWordList[3] < maxProbKeyWordList[2]:
				execute = False
			if execute:
				(x, y, w, h) = wordList[maxProbKeyWordListInd[k]].dims
				cv2.rectangle(imgKeyWords, (x, y), (x + w, y + h), (0, 255, 0), 2)
				for i in range(len(keyWordListInd[k])): #characters in keyword
					label = keyWordList[k][i]
					charInd = wordList[maxProbKeyWordListInd[k]].charList[i]
					(xC, yC, wC, hC) = boxes[charInd]
					cv2.putText(imgKeyWords, label, (xC - 10, yC - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
		imageS = self.ResizeImage(imgKeyWords, 800, 800)
		cv2.imshow("Keywords Image", imageS)
		cv2.waitKey(0)

	def OCRHandwriting(self):
		for image_file in os.listdir(self.InputDir):
			# load the input file from disk
			image_path = self.InputDir + '\\' + image_file
			image = cv2.imread(image_path)
			if type(image) is np.ndarray:  # only process if image file
				image = self.ResizeImage(image, 2000, 2000)
				gray = self.Preprocess(image) #preprocess the image, convert to gray
				#imageS = self.ResizeImage(gray, 800, 800)
				#cv2.imshow("Keywords Image", imageS)
				#cv2.waitKey(0)
				chars, wordList = self.FindCharsWords(gray) #find characters and words, store as [image, (x, y, w, h)]
				self.RunModel(chars, wordList, gray, image) #run the model to predict characters


class Word:
	dims = (0,0,0,0)  #= np.zeros(4, dtype=int) #x,y,w,h
	charList = [] #list of character indices
	avgCharSpac: int = 0 #initialize average character spacing to -1
	#def __init__(self):
		#self.dims = (0,0,0,0)

class keyWord:
	Chars = [] #list of characters in keyword, caps [0 to n keywords - 1]
	CharsInd = [] #indices of characters from 0 to 35, [0 to n keywords - 1]
	NInstances: int  #number of instances of keyword to search for, integer
	MaxProb = ()  #maximum probability of max prob word in wordList matching keyWord, [0 to nInstances - 1]
	MaxProbWordInd = () #wordList index corresponding to maxProb

	def __init__(self, chars: list, nInstances: int, labelNames: list):
		self.NInstances = nInstances
		self.Chars = chars
		self.MaxProb = (0,0)
		self.MaxProbWordInd = (-1,-1)
		self.CharsInd = self.assignCharsInd(chars, labelNames)

	def assignCharsInd(self, chars: list, labelNames: list):
		keyWordInd = []
		for char in chars:
			for i in range(len(labelNames)):
				l = labelNames[i]
				if l == char:
					keyWordInd.append(i)
					break
		return keyWordInd


