# modified from: https://www.pyimagesearch.com/2020/08/24/ocr-handwriting-recognition-with-opencv-keras-and-tensorflow/
# identify characters within an image
# import the necessary packages
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
from pickle import load
import numpy as np
import argparse
import imutils
import cv2
import os
import h5py
import csv

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

	InputDir: str  # image input directory

	def __init__(self, inputdir: str):
		self.InputDir = inputdir

	def OCRHandwriting(self):
		for image_file in os.listdir(self.InputDir):
			# load the input file from disk
			image_path = self.InputDir + '\\' + image_file
			image = cv2.imread(image_path)
			if type(image) is np.ndarray:  # only process if image file
				# create new image object
				wbimage = WBImage(self.InputDir)
				wbimage.image = wbimage.ResizeImage(image, 2000, 2000)
				wbimage.Preprocess()  # preprocess the image, convert to gray
				# imageS = self.ResizeImage(gray, 800, 800)
				# cv2.imshow("Keywords Image", imageS)
				# cv2.waitKey(0)
				wbimage.FindCharsWords()
				# find characters and words, store as [image, (x, y, w, h)]
				wbimage.RunModel(image_file)  # run the model to predict characters

class WBImage:
	# declare class variables
	InputDir: str  # image input directory
	Num_AZ_Model: object  # input neural network model for predicting most likely a-z and 0 - 9 character combined
	Num_Model: object # input neural network model for predicting most likely 0 - 9 character combined
	LabelNames = [l for l in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
	charList: list # list of Character class objects
	wordList: list # list of Word class objects
	keyWordList: list # list of keyWord class objects
	image: object #image in colour, scaled with aspect ratio to max 2000 x 2000
	gray: object #image in grayscale, scaled with aspect ratio to max 2000 x 2000
	depthFrom: str #depth from found for the current image, in string format
	depthTo: str #depth to found for the current image, in string format


	# CharList: list #list of Character class objects found in image by neural network
	# WordList: list #list of Word class objects comprising 2 or more characters nearby
	# ProbNumList: list  # list corresponding to wordList indices with probability of number
	# WordCharList: list  # list of character arrays in words corresponding to wordList
	# KeyWordList: list  # list of char lists defining keywords

	def __init__(self, inputdir: str):
		self.InputDir = inputdir
		self.Num_AZ_Model = load_model('number_az_model.h5')
		self.Num_Model = load_model('mnist_number_model.h5')
		self.charList = []
		self.wordList = []
		self.BuildKeyWordList()

	def BuildKeyWordList(self):
		self.keyWordList = []
		self.keyWordList.append(KeyWord(['F', 'R', 'O', 'M'], 1, self.LabelNames))
		self.keyWordList.append(KeyWord(['T', 'O'], 1, self.LabelNames))
		self.keyWordList.append(KeyWord(['D', 'E', 'P', 'T', 'H'], 2, self.LabelNames))  # allow for two depths to be found
		self.keyWordList.append(KeyWord(['D', 'R', 'Y'], 1, self.LabelNames))
		self.keyWordList.append(KeyWord(['W', 'E', 'T'], 1, self.LabelNames))

	def ResizeImage(self, img, maxW, maxH):
		# resizes the image based on the maximum width and maximum height, returns the resized image
		if img.ndim == 2:  # black and white
			(tH, tW) = img.shape
		else:  # colour
			(tH, tW, tmp) = img.shape
		# if the width is greater than the height (usually will be), resize along the width dimension
		if tW / tH > maxW / maxH:  # width is the defining dimension along which to resize
			hCheck = tH / (tW / maxW)
			if hCheck > 1.00001:  # check not < 1, will throw error
				img = imutils.resize(img, width=maxW)
			else:
				img = imutils.resize(img, width=maxW, height=1)
		# otherwise, resize along the height
		else:
			wCheck = tW / (tH / maxH)
			if wCheck > 1.00001:  # check not < 1, will throw error
				img = imutils.resize(img, height=maxH)
			else:
				img = imutils.resize(img, height=maxH, width=1)
		return img

	def Preprocess(self):
		# convert image to grayscale, and blur it to reduce noise
		self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		self.gray = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		# gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)[1]
		# gray = cv2.GaussianBlur(gray, (5, 5), 0)
		# Applied dilation
		kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
		self.gray = cv2.morphologyEx(self.gray, cv2.MORPH_ERODE, kernel3)
		# grayS = self.ResizeImage(gray, 800, 800)
		# cv2.imshow("Preprocessed", grayS)
		# cv2.waitKey(0)
		#return gray

	def FindCharsWords(self):
		# perform edge detection, find contours in the edge map, and sort the
		# resulting contours from left-to-right, find words
		# fac = 2 #factor by which to temporarily upscale images to improve edge detection
		# tH, tW = gray.shape
		# imgTmp = self.ResizeImage(gray, tW * fac, tH * fac) #temporarily upsize by fac to make edge detection work better

		edged = cv2.Canny(self.gray, 30, 150)
		# cv2.imshow("padded", edged)
		# cv2.waitKey(0)
		cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # cv2.CHAIN_APPROX_NONE)
		cnts = imutils.grab_contours(cnts)
		cnts = sort_contours(cnts, method="left-to-right")[0]
		# initialize the list of contour bounding boxes and associated
		# cnts = cnts / fac
		# characters that we'll be OCR'ing
		# chars = []  # pairs of character images and dimensions
		# loop over the contours and populate if they pass the criteria
		self.ProcessChars(cnts)
		return

	def ProcessChars(self, cnt):
		# function takes in image and contour and filters the characters to those within words,
		# and those with appropriate sizes, and adjusts white space
		#charList = []  # pairs of character images and dimensions
		# charDict = {} #charID, ndarray[28x28x1] image, (x,y,w,h) of original image
		for i in range(len(cnt)):
			c = cnt[i]
			# compute the bounding box of the contour
			(x, y, w, h) = cv2.boundingRect(c)
			# filter out bounding boxes, ensuring they are neither too small
			# nor too large
			# if (w >= 5 and w <= 150) and (h >= 8 and h <= 120):
			if (w >= 5 and w <= 375) and (h >= 5 and h <= 300) and w / h < 22 and h / w < 22:

				# extract the character and threshold it to make the character
				# appear as *white* (foreground) on a *black* background, then
				# grab the width and height of the thresholded image
				roi = self.gray[y:y + h, x:x + w]
				thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

				thresh = self.ResizeImage(thresh, 22, 22)  # resize the image

				# re-grab the image dimensions (now that its been resized)
				# and then determine how much we need to pad the width and
				# height such that our image will be 28x28
				(tH, tW) = thresh.shape
				dX = int(max(6, 28 - tW) / 2.0)
				dY = int(max(6, 28 - tH) / 2.0)
				# pad the image and force 28x28 dimensions
				padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX,
											borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
				padded = cv2.resize(padded, (28, 28))
				# cv2.imshow("padded", padded)
				# cv2.waitKey(0)
				# print("{}, {}".format(w, h))
				# prepare the padded image for classification via our
				# handwriting OCR model
				padded = padded.astype("float32") / 255.0
				padded = np.expand_dims(padded, axis=-1)
				# check if small character, unlikely text (could be punctuation):
				smallCharFilt = False
				if h < 22:
					smallCharFilt = True
				self.charList.append(Character(padded, (x, y, w, h), False, smallCharFilt,
										  False))  # update our list of characters that will be OCR'd
			# charDict[i] = (padded, (x, y, w, h))

		# check each character to make sure not overlapping with an another character, discard if so
		# removeList = []
		removeList = self.CheckOverlap()
		for i in removeList:
			self.charList[i].OvlFilt = True
		# for i in range(len(removeList)-1,-1,-1):
		# del chars[removeList[i]]

		#wordList = []
		# now loop through chars and perform checks, assign to words
		for i, char in enumerate(self.charList):
			if not char.OvlFilt:
				fndWord = False
				(x, y, w, h) = char.Dims  # read in character dimensions
				for j, word in enumerate(self.wordList):  # loop through existing word list
					prevChar = self.charList[word.charList[len(word.charList) - 1]]  # read in previous character in word
					fndWord = self.CharChecks(i, char, prevChar, j, word)
					if fndWord:
						(xW, yW, wW, hW) = word.dims  # read in word dimensions
						# update wordList parameters
						yWN = min(y, yW)
						hWN = max(y + h, yW + hW) - yWN
						wWN = x + w - xW
						word.charList.append(i)
						word.dims = (xW, yWN, wWN, hWN)
						# assign avg word spacing
						if len(word.charList) > 2:
							xxx = 1
						word.avgCharSpac = (word.avgCharSpac * (len(word.charList) - 2) + max(x - (xW + wW), 0)) / (
									len(word.charList) - 1)
						word.avgCharH = (word.avgCharH * (len(word.charList) - 1) + h) / len(word.charList)
						word.avgCharW = (word.avgCharW * (len(word.charList) - 1) + w) / len(word.charList)
						char.InWord = True
						break  # exit for loop
				if not (fndWord):  # start a new word
					newWord = Word()
					newWord.dims = char.Dims
					newWord.charList = [i]
					self.wordList.append(newWord)
					char.InWord = True
				# wordList.append((char[1],[i])) #add dimensions of first character and charID to the wordlist

		# final loop to throw out words that only have one character
		for i in range(len(self.wordList) - 1, -1, -1):
			words = self.wordList[i]
			if len(words.charList) < 2:
				self.charList[words.charList[0]].InWord = False  # change back to false, no longer in a word
				del self.wordList[i]
		# else:#check for overlapping characters and remove from word if so
		# removeList = self.CheckOverlap(chars)
		# for i in range(len(removeList) - 1, -1, -1):
		# del chars[removeList[i]]

		#return charList, wordList

	def CharChecks(self, i, char, prevChar, j, word):
		# function runs a series of checks to check whether character 'char' is in word 'word', returns true is so, or false if not
		(x, y, w, h) = char.Dims  # read in character dimensions
		(xW, yW, wW, hW) = word.dims  # read in word dimensions
		(xC, yC, wC, hC) = prevChar.Dims  # read in last character dimensions in word
		# compare to determine whether character is part of current word
		xDif = x - (xW + wW)  # check x
		if xDif >= hW/1.2:  # compare to word height to check if close enough to word to be included (i.e. whitespace between)
			return False
		# check y-overlap against previous character in word (instead of fill word
		if y > (yC + hC) or (y + h) < yC:  # need to also check amount of overlap
			return False
		ovl = (min(y + h, yC + hC) - max(y, yC)) / (max(y + h, yC + hC) - min(y, yC))  # percentage overlap
		if ovl <= 0.3:  # set 30% overlap threshold
			return False
		htRatio = h / hW
		if htRatio > 2.5 or htRatio < 0.3:  # thresholds for height ratios
			return False
		if not (len(word.charList) <= 3 or (len(word.charList) > 2 and (max(xDif, 0) - word.avgCharSpac) / hW < .4)):
			#check space between characters relative to mean space
			return False
		# final check - look for a change in average spacing between characters in a word
		else:
			return True

	def CheckOverlap(self):
		# check each character to make sure not overlapping with an another character
		# chars [ndarray[28x28x1], (x,y,w,h)] - list of characters
		removeList = []
		for i, charI in enumerate(self.charList):
			# charI = chars[i]
			(x, y, w, h) = charI.Dims  # read in character dimensions
			discard: bool = False
			for j, charJ in enumerate(self.charList):
				if j == i:
					continue
				# charJ = chars[j]
				(xC, yC, wC, hC) = charJ.Dims  # read in character dimensions
				# xFuz = wC * 0.1 #if over 60% overlap in x and y then remove
				# yFuz = hC * 0.1
				if w * h <= wC * hC:  # only discard the smaller of the two
					if ((x > xC and x < xC + wC) or (x + w > xC and x + w < xC + wC)) and (
							(y > yC and y < yC + hC) or (y + h > yC and y + h < yC + hC)):
						# if x > xC - xFuz and x + w < xC + wC + xFuz and y > yC - yFuz and y + h < yC + hC + yFuz:
						# there is overlap, determine how much as proportion of smaller item
						aOvl = (min(x + w, xC + wC) - max(x, xC)) * (min(y + h, yC + hC) - max(y, yC))
						percOvl = aOvl / (w * h)
						if percOvl > 0.6:  # overlap > 60%
							removeList.append(i)
							break
		return removeList

	def RunModel(self, image_file):
		# run the model to predict characters
		#image_file is the filename

		# function level variables
		# define the list of label names
		imgAnno = self.image.copy()  # make a local copy of coloured image
		#probNumList = []  # list corresponding to wordList indices with probability of number
		#wordCharList = []  # list of character arrays in words corresponding to wordList

		dryInd = 3
		wetInd = 4
		keywordProbMin = 0.4 #if < 40% then ignore

		# extract the bounding box locations and padded characters
		# boxes = [b[1] for b in chars]
		# chars = np.array([c[0] for c in chars], dtype="float32")
		boxes = [b.Dims for b in self.charList]
		imageData = np.array([c.Data for c in self.charList], dtype="float32")
		# OCR the characters using our handwriting recognition model
		preds = self.Num_AZ_Model.predict(imageData) #predict most likely num and char
		predsNum = self.Num_Model.predict(imageData) #predict most likely num

		# loop over the predictions and bounding box locations together
		for n, (pred, (x, y, w, h)) in enumerate(zip(preds, boxes)):
			# for (x, y, w, h) in boxes:
			# find the index of the label with the largest corresponding
			# probability, then extract the probability and label
			i = np.argmax(pred)
			prob = pred[i]
			label = self.LabelNames[i]
			# draw the prediction on the image
			# print("[INFO] {} - {:.2f}%".format(label, prob * 100))
			cv2.rectangle(imgAnno, (x, y), (x + w, y + h), (0, 255, 0), 2)
			if self.charList[n].OvlFilt == False and self.charList[n].SmallFilt == False:  # only add text to the image if filter out flag is false
				cv2.putText(imgAnno, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
		# loop over words, draw rectangle around
		for wInd, words in enumerate(self.wordList):
			# if len(words[1]) > 1: #show wordBoxes if > 1 character
			(x, y, w, h) = words.dims
			cv2.rectangle(imgAnno, (x, y), (x + w, y + h), (255, 51, 204), 2)  # rectangle around word

		# show the image
		imageS = self.ResizeImage(imgAnno, 800, 800)
		cv2.imshow("Image", imageS)
		# grayS = self.ResizeImage(gray, 800, 800)
		# cv2.imshow("Gray", grayS)
		cv2.waitKey(0)

		# loop over words again
		for wInd, words in enumerate(self.wordList):
			probNum = 0  # probability that word contains a number
			wordChars = []  # list of character indices in word
			#wordCharsText = [] #list of characters in word
			# loop over characters in word and determine probability of #,
			for i in words.charList:  # for each ith character
				prob = preds[i]
				for j in range(10):  # loop through probability of number characters
					probNum += prob[j]
				# probNum /= 10
				wordChars.append(self.LabelNames[np.argmax(preds[i])])  # store the max likelihood character in wordCharList
				#wordCharsText.append

			probNum /= len(words.charList)
			words.probNum = probNum
			words.wordCharList = wordChars
			#probNumList.append(probNum)
			#wordCharList.append(wordChars)

			# check probability of being a keyword
			for k in self.keyWordList:
				keyWordInd = k.CharsInd
				probKeyWord = 0
				if len(words.charList) == len(keyWordInd):  # same length, so check probability
					for i in range(len(keyWordInd)):  # for each ith character
						ind = words.charList[i]
						probKeyWord += preds[ind][keyWordInd[i]]  # find probability that ith characters match
					probKeyWord /= len(keyWordInd)
					# now compare to list, and replace if the new highest likelihool of word is found
					for n in range(len(k.MaxProb)):
						if probKeyWord > k.MaxProb[n]:  # new highest likelihood found
							k.MaxProb[n] = probKeyWord
							k.MaxProbWordInd[n] = wInd  # corresponding word index
							break
		# show image of keyWords picked out, and print probabilities correct
		imgKeyWords = self.image.copy()
		for ii, k in enumerate(self.keyWordList):
			for n in range(len(k.MaxProb)):
				temp: str = ""
				for c in k.Chars:
					temp = temp + c
				temp = temp + ": P={:.1f}%".format(k.MaxProb[n] * 100)
				print(temp)
				execute = True
				if k.MaxProb[n] < keywordProbMin:  # keyword not found with sufficient probability (40%)
					execute = False
				# hardcode for DRY vs WET
				if ii == dryInd and self.keyWordList[dryInd].MaxProb[n] < self.keyWordList[wetInd].MaxProb[n]:
					# compare probability of dry vs wet, only show the higher probability
					execute = False
				elif ii == wetInd and self.keyWordList[wetInd].MaxProb[n] < self.keyWordList[dryInd].MaxProb[n]:
					execute = False
				if execute:
					(x, y, w, h) = self.wordList[k.MaxProbWordInd[n]].dims
					cv2.rectangle(imgKeyWords, (x, y), (x + w, y + h), (0, 255, 0), 2)
					for i in range(len(k.Chars)):  # characters in keyword
						label = k.Chars[i]
						charInd = self.wordList[k.MaxProbWordInd[n]].charList[i]
						(xC, yC, wC, hC) = boxes[charInd]
						cv2.putText(imgKeyWords, label, (xC - 10, yC - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255),2)
		imageS = self.ResizeImage(imgKeyWords, 800, 800)
		cv2.imshow("Keywords Image", imageS)
		cv2.waitKey(0)

		# TEMPORARY - CREATE Training Set by saving photos to folders
		valid = False
		inp: str
		dataList = []
		labelList = []
		punctIDList = []
		while not valid:
			inp = input("Add to training set? (y/n): ")
			if inp == "": inp = "Y"
			inp = inp.upper()
			if inp == "Y" or inp == "N":
				valid = True
		if inp == "Y":  # write image to training set folder
			image_path = self.InputDir + '\\' + image_file
			imageTemp = cv2.imread(image_path)
			image_path_out = self.InputDir + '\\trainset\\' + image_file
			cv2.imwrite(image_path_out, imageTemp)
		# label data
		# write labelled data to file

		#if inp == "Y":
			# build and scale feature matrix for words in current image, store in dataList and labelList
			(tH, tW) = self.gray.shape
			dataList, labelList, punctIDList = self.BuildFeatureMatrix(True, tH, tW, imgKeyWords, keywordProbMin)
			#self.SaveUpdateTrainingSet(r'input/depth_train_dataset.hdf5', image_file, dataList, labelList)
			self.SaveUpdateTrainingSetCSV('depth_train_dataset.csv', image_file, dataList, labelList)
		else: #input = "N"
			(tH, tW) = self.gray.shape
			#dataList is size #words x n features (=5)
			#label list is size #words x 1 (1 or 0)
			dataList, labelList, punctIDList = self.BuildFeatureMatrix(False, tH, tW, imgKeyWords,keywordProbMin)
		y_result = self.RunWordNumSVMModel(dataList) #output from 1 to 0 representing likelihood of being a number
		yMaxInd = [-1,-1] #word index corresponding to highest and second highest y_result
		yMax = [0,0] #highest and second highest y_result
		for ind, (word, y) in enumerate(zip(self.wordList, y_result)):
			output = "".join(word.wordCharList)
			output += ": {:.2f}".format(y[1])
			print(output)

			#check if most or second most likely, store if so
			for n in range(len(yMaxInd)):
				if y[1] > yMax[n]:
					yMax[n] = y[1]
					yMaxInd[n] = ind
					break

		#output results to image
		for n, p in zip(yMaxInd, yMax): #loop through 2 number words found...
			if p > 0.2: #only output if prob of word being number > 20%
				(x, y, w, h) = self.wordList[n].dims
				cv2.rectangle(imgKeyWords, (x, y), (x + w, y + h), (0, 0, 255), 2)
				# check for punctuation, add to image if it exists
				if punctIDList[n] > -1:  # -1 is null val for no punctuation
					(xP, yP, wP, hP) = self.charList[punctIDList[n]].Dims
					cv2.putText(imgKeyWords, ".", (xP - 10, yP - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
				for c in self.wordList[n].charList:  # number characters, pull labels from numeric neural network
					(xC, yC, wC, hC) = self.charList[c].Dims
					i = np.argmax(predsNum[c])
					prob = predsNum[i]
					label = self.LabelNames[i]
					cv2.putText(imgKeyWords, label, (xC - 10, yC - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
		imageS = self.ResizeImage(imgKeyWords, 800, 800)
		cv2.imshow("Final Image", imageS)
		cv2.waitKey(0)



	def BuildFeatureMatrix(self, labeldata: bool, tH, tW, image, keywordProbMin):
		# built feature matrix
		#parameters:
		#labeldata: True if interactive user labelling data
		# features:
		# x_dist - x distance from nearest keyword, scaled to [-1,1] by bounds of whiteboard image
		# y_dist - y distance from nearest keyword, scaled to [-1,1] by bounds of whiteboard image
		# p_numb - average likelihood that characters in word are numeric [0,1]
		# punct - whether or not the word contains a '.' punctuation character, [0=False,1=True]
		# num_chars - #number of characters scaled ###number of characters, [0,1], 1 if 3-6 characters, 0 otherwise
		featnames = ["x_dist", "y_dist", "prob_numb", "punct", "num_chars"]
		n_feat = 5
		data = [] #np.zeros(len(wordList), n_feat)
		labels = [] #1 for depth value word, 0 for not. only relevant if labeldata = true, otherwise blank
		punctIDList = [] #character ID of punction for each word, -1 if no punctuation
		# obtain feature vector for each word
		for n, word in enumerate(self.wordList):

			#img = image.copy()
			#(x, y, w, h) = word.dims
			#cv2.rectangle(img, (x, y), (x + w, y + h), (0, 176, 240), 2)
			#imageS = self.ResizeImage(img, 800, 800)
			#cv2.imshow("Keywords Image", imageS)
			#cv2.waitKey(0)

			#ex_param = np.zeros(1, n_feat)
			x_dist, y_dist = self.FindClosestKeyword(word, tH, tW, keywordProbMin)
			p_numb = word.probNum
			punct, punctID = self.FindPunctuation(word) #punct is true or false, and if true, return charID for punctuation
			punctIDList.append(punctID)
			num_chars: float = len(word.charList) / 10 #scale / 10, assume 10 is max reasonable # of characters
			"""
			if len(word.charList) <= 6 and len(word.charList) >= 3:
				num_chars = 1
			else:
				num_chars = 0
			"""
			datarow = np.array([x_dist, y_dist, p_numb, punct, num_chars], dtype="float32")
			#datarow = np.zeros(1, n_feat, dtype="float32")
			data.append(datarow)
			if labeldata: #interactive user data labelling
				#first, print vector to screen:
				output = ("WORD #{}: ".format(n))
				output += "".join(word.wordCharList)
				print(output)
				for i, name in enumerate(featnames):
					output = name + ": {:.2f}".format(datarow[i])
					print(output)

				img = image.copy()
				(x, y, w, h) = word.dims
				cv2.rectangle(img, (x, y), (x + w, y + h), (0, 176, 240), 2)
				imageS = self.ResizeImage(img, 800, 800)
				cv2.imshow("Keywords Image", imageS)
				cv2.waitKey(0)
				valid = False
				inputval: int = 0
				while not valid:
					inp = input("Depth value (enter 1) or not (enter 0):")
					if inp == "": inp = "0"
					if inp == "1" or inp == "0":
						valid = True
						inputval = int(inp)
				labels.append(inputval)


		return data, labels, punctIDList

	def FindClosestKeyword(self, word, tH, tW, keywordProbMin):
		# finds and returns x and y distance to closest keyword
		# searching keywords index 0 - 2
		(x, y, w, h) = word.dims
		xc = x + w / 2
		yc = y + h / 2
		xMinDif: float = tW  # minimum x and y difference between word center and keyword center, initialize to image dimensions
		yMinDif: float = tH
		hMinDif: float = tW**2 + tH**2 #minimum hypoteneuse
		for k, keyWord in enumerate(self.keyWordList):
			if k >= 3: break  # exit loop if past 'depth', hardcoded
			for n, p in zip(keyWord.MaxProbWordInd, keyWord.MaxProb):
				if p < keywordProbMin: continue #ignore keyword if < min cutoff, currently 40%
				(xK, yK, wK, hK) = self.wordList[n].dims
				xcK = xK + wK / 2
				ycK = yK + hK / 2
				xdif = xc - xcK
				ydif = yc - ycK
				hdif = xdif**2 + ydif**2
				if abs(hdif) < abs (hMinDif):
					hMinDif = hdif #assign if closer
					xMinDif = xdif
					yMinDif = ydif
				#if abs(xdif) < abs(xMinDif): xMinDif = xdif  # assign if closer
				#if abs(ydif) < abs(yMinDif): yMinDif = ydif  # assign if closer
		# scale
		xMinDif /= tW
		yMinDif /= tH
		return xMinDif, yMinDif

	def FindPunctuation(self, word):
		#function looks for punctuation sized character within word and returns true if found, false if not
		(xW, yW, wW, hW) = word.dims
		fndPunct: bool = False
		punctID: int = -1
		for ind, char in enumerate(self.charList):
			if char.InWord == True or char.SmallFilt == False: #character is not in a word, and has been flatted as a small character, possible punctuation
				continue
			(x, y, w, h) = char.Dims
			# check within lower half of word, expanded downward by 1/3 of word height
			if x < xW + (wW*0.1) or x > xW + wW - (wW*0.1): #search middle 80% of word for punctuation
				continue #not within x
			if y < yW + hW/2 or y > yW + hW + hW/4:
				continue #not within reasonable y
			fndPunct = True
			punctID = ind
			break
		return float(fndPunct), punctID

	def SaveUpdateTrainingSetCSV(self, fname, imagefile, dataList, labelList):
		# function checks if an existing training set exists, and builds a new one if not
		# structure of CSV:
		# row 0 = labels
		# rows 1 - 5 = data
		# row 6 = image file name
		# check if file exists:
		exists = os.path.exists(fname)
		writemode = 'w'
		if exists: writemode = 'a'
		with open(fname, writemode) as csvfile:
			filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')
			if not exists:
				filewriter.writerow(["labels", "x_dist", "y_dist", "prob_numb", "punct", "num_chars", "filename"]) #header row first
			for dataLine, labelLine in zip(dataList, labelList):
				outputlist = []
				outputlist.append(str(labelLine))
				for a in dataLine:
					outputlist.append(str(a))
				outputlist.append(imagefile) #image file name
				filewriter.writerow(outputlist)

	def RunWordNumSVMModel(self, dataList):
		#function runs the SVM on the word dataset to classify if word is a number (1) or not (0)

		# load the model
		model = load(open('word_number_model_prob.pkl', 'rb'))
		# load the scaler
		scaler = load(open('word_number_scaler_prob.pkl', 'rb'))

		#scale the data using saved scaler
		data = np.array(dataList, dtype="float32")
		data = scaler.transform(data)

		yval = model.predict(data) #1 if number, 0 if not
		yval_prob = model.predict_proba(data)

		return yval_prob

	def SaveUpdateTrainingSet(self, fname, imagefile, dataList, labelList):
		#FUNCTION NOT WORKING, not currently used
		#function checks if an existing training set exists, and builds a new one if not
		#check if file exists
		exists = os.path.exists(fname)
		dataAll: list  # = []
		labalAll: list  # = []
		f: h5py.File
		if exists:
			f = h5py.File(fname, 'r+')
			try:
				dataRead = f['data'][:]
				labelRead = f['labels'][:]
				dataAll = list(dataRead)
				labelAll = list(labelRead)
			except:
				xxx = 1
		else: #create new file
			f = h5py.File(fname, 'w')

		#blnRead: bool
		#try: #read all the data in

			#blnRead = True
		#except:#file does not exist
			#blnRead = False
		#if blnRead:
			#dataAll = list(dataRead)
			#labelAll = list(labelRead)
			#for dataLine in dataRead:
				#dataAll.append(dataLine)
			#for labalLine in labelRead:
				#labelAll.append(labelsRead)
		dataAll2 = np.array(dataList, dtype="float32")
		labelAll2 = np.array(labelList, dtype="int")
		f.create_dataset('data', data=dataAll2)
		f.create_dataset('labels', data=labelAll2)

		f.close()


class Word:
	dims: tuple  # = (0, 0, 0, 0)  #= np.zeros(4, dtype=int) #x,y,w,h
	charList = []  # list of character indices
	avgCharSpac: float = 0  # initialize average character spacing to -1
	avgCharW: float = 0
	avgCharH: float = 0
	probNum: float = 0
	wordCharList: list = [] #most likely word characters based on NN model output



class Character:
	Data: np.zeros(shape=(28, 28, 1))  # b&W shade data, 28x28 size image
	Dims: tuple  # x,y,w,h
	OvlFilt = False  # true flags overlapping characters to exclude from certain operations
	SmallFilt = False
	InWord = False  # true if assigned to a word, false if not

	def __init__(self, data: np.array, dims: tuple, ovlfilt: bool, smallfilt: bool, inword: bool):
		self.Data = data
		self.Dims = dims
		self.OvlFilt = ovlfilt
		self.SmallFilt = smallfilt
		self.InWord = inword


class KeyWord:
	Chars: list  # list of characters in keyword, caps [0 to n keywords - 1]
	CharsInd: list  # indices of characters from 0 to 35, [0 to n keywords - 1]
	NInstances: int  # number of instances of keyword to search for, integer
	MaxProb: list  # maximum probability of max prob word in wordList matching keyWord, [0 to nInstances - 1]
	MaxProbWordInd: list  # wordList index corresponding to maxProb

	def __init__(self, chars: list, nInstances: int, labelNames: list):
		self.NInstances = nInstances
		self.Chars = chars
		self.MaxProb = [0.0] * nInstances  # (0 for n in range(nInstances))
		self.MaxProbWordInd = [-1] * nInstances  # (-1 for n in range(nInstances))
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
