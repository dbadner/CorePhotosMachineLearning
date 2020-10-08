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
			img = imutils.resize(img, width=maxW)
		# otherwise, resize along the height
		else:
			img = imutils.resize(img, height=maxH)
		return img

	def Preprocess(self, image):
		# convert image to grayscale, and blur it to reduce noise
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		# gray = cv2.GaussianBlur(gray, (5, 5), 0)
		# Applied dilation
		kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		gray = cv2.morphologyEx(gray, cv2.MORPH_ERODE, kernel3)
		#cv2.imshow("Preprocessed", gray)
		#cv2.waitKey(0)
		return gray

	def FindCharsWords(self, gray):
		# perform edge detection, find contours in the edge map, and sort the
		# resulting contours from left-to-right, find words
		edged = cv2.Canny(gray, 30, 150)
		cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		cnts = sort_contours(cnts, method="left-to-right")[0]
		# initialize the list of contour bounding boxes and associated

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
			if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):

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

		wordList = []
		#now loop through chars and assign to words
		for i in range(len(chars)):
			char = chars[i]
			(x, y, w, h) = char[1] #read in character dimensions
			#charDict[i] = char #store in dictionary
			fndWord = False
			for j in range(len(wordList)): #loop through existing word list
				word = wordList[j]
				(xW, yW, wW, hW) = word[0] #read in word dimensions
				#compare to determine whether character is part of current word
				xDif = x - (xW+wW) #check x
				if xDif < hW/2: #compare to word height
					#check y-overlap
					if not(y > (yW + hW) or (y + h) < yW):#need to also check amount of overlap
						ovl = (min(y+h,yW+hW) - max(y,yW)) / (max(y+h,yW+hW) - min(y,yW)) #percentage overlap
						if ovl > 0.3: #set 30% overlap threshold
							htRatio = h / hW
							if htRatio < 2.5 and htRatio > 0.3: #thresholds for height ratios
								fndWord = True
								#update wordList parameters
								yWN = min(y,yW)
								hWN = max(y+h, yW+hW) - yWN
								wWN = x + w - xW
								word[1].append(i)
								#word[0] = (xW, yWN, wWN, hWN)
								wordList[j] = (((xW, yWN, wWN, hWN), word[1]))
								continue  # exit for loop
			if not(fndWord): #start a new word
				wordList.append((char[1],[i])) #add dimensions of first character and charID to the wordlist
		# final loop to throw out chars that are not part of words (i.e. wordList entries with only 1 char)

		return chars, wordList

	def OCRHandwriting(self):
		for image_file in os.listdir(self.InputDir):
			# load the input file from disk
			image_path = self.InputDir + '\\' + image_file
			image = cv2.imread(image_path)
			if type(image) is np.ndarray:  # only process if image file
				image = self.ResizeImage(image, 800, 800)
				gray = self.Preprocess(image) #preprocess the image, convert to gray
				chars, wordList = self.FindCharsWords(gray) #find characters and words, store as [image, (x, y, w, h)]

				# extract the bounding box locations and padded characters
				boxes = [b[1] for b in chars]
				chars = np.array([c[0] for c in chars], dtype="float32")
				# OCR the characters using our handwriting recognition model
				preds = self.Model.predict(chars)
				# define the list of label names
				labelNames = "0123456789"
				labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
				labelNames = [l for l in labelNames]

				# loop over the predictions and bounding box locations together
				for (pred, (x, y, w, h)) in zip(preds, boxes):
					# for (x, y, w, h) in boxes:
					# find the index of the label with the largest corresponding
					# probability, then extract the probability and label
					i = np.argmax(pred)
					prob = pred[i]
					label = labelNames[i]
					# draw the prediction on the image
					print("[INFO] {} - {:.2f}%".format(label, prob * 100))
					cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
					cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
				#loop over words and show wordBoxes if > 1 character
				for words in wordList:
					if len(words[1])>1:
						(x,y,w,h) = words[0]
						cv2.rectangle(image, (x, y), (x + w, y + h), (255, 51, 204), 2)
				# show the image
				cv2.imshow("Image", image)
				cv2.imshow("Gray", gray)
				cv2.waitKey(0)
