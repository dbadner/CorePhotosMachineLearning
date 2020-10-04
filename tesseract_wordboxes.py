# import the necessary packages
from PIL import Image
from pytesseract import Output
import pytesseract
import argparse
import cv2
import os
import numpy as np

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'
TESSDATA_PREFIX = 'C:/Program Files/Tesseract-OCR'


class TessFindWords:
    # class variables
    InputDir: str  # input directory path with photographs
    # custom options
    CustomConfig: str

    # parameterized constructor
    def __init__(self, inputdir: str):
        self.InputDir = inputdir
        self.CustomConfig = '--psm 11'

    def RunTess(self, showimages: bool, thresh: bool):  # , wbcrop: dict):
        # construct the argument parse and parse the arguments
        # ap = argparse.ArgumentParser()
        # hack temporary hard code for debugging
        # ap.add_argument("-i", "--image", required=False, default="input/17316.png",
        #                help="path to input image to be OCR'd")
        # ap.add_argument("-p", "--preprocess", type=str, default="stdout",
        #                help="type of preprocessing to be done")
        # args = vars(ap.parse_args())

        for image_file in os.listdir(self.InputDir):
            image_path = self.InputDir + '\\' + image_file

            img = cv2.imread(image_path)

            exppix = 10 # of pixels by which to expand boxes around words

            if type(img) is np.ndarray:  # only process if image file

                # cropdims: np.ndarray = wbcrop[image_file]

                # gray = cv2.cvtColor(img[cropdims[1].astype(int):cropdims[3].astype(int), cropdims[0].astype(int):cropdims[2].astype(int)], cv2.COLOR_BGR2GRAY)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                grayh, grayw = gray.shape
                # show the input image
                # if showimages: cv2.imshow("Input Image", img)

                # check to see if we should apply thresholding to preprocess the
                # image
                if thresh:
                    #gray = cv2.medianBlur(gray, 3)
                    #ret, gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                    #gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
                    ## Applied dilation
                    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    gray = cv2.morphologyEx(gray, cv2.MORPH_ERODE, kernel3)
                # make a check to see if median blurring should be done to remove
                # noise
                # elif args["preprocess"] == "blur":


                # write the grayscale image to disk as a temporary file so we can
                # apply OCR to it
                filename = "{}.png".format(os.getpid())
                cv2.imwrite(filename, gray)

                # load the image as a PIL/Pillow image
                imgpillow = Image.open(filename)
                text = pytesseract.image_to_string(imgpillow, config=self.CustomConfig)
                #target = pytesseract.image_to_string(image, lang='eng', boxes=False, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')

                print(text)
                file1 = open(r"output/" + image_file + "TessOutImgToStr.txt", "w+")
                file1.write(text)
                file1.close()

                d = pytesseract.image_to_data(imgpillow, output_type=Output.DICT, config=self.CustomConfig)
                n_boxes = len(d['left'])
                for i in range(n_boxes):
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # show the output image
                # Read image
                # cv2.namedWindow("Output", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
                # imS = cv2.resize(gray, (960, 540))  # Resize image
                if showimages: cv2.imshow("Output Data Image", gray)  # Show image

                # write text output
                file2 = open(r"output/" + image_file + "TessOutImgToData.txt", "w+")

                #get character boxes
                #h, w, c = img.shape
                boxes = pytesseract.image_to_boxes(imgpillow, config=self.CustomConfig) #, output_type=Output.DICT, config=self.CustomConfig)
                for b in boxes.splitlines():
                    b = b.split(' ')
                    img = cv2.rectangle(img, (int(b[1]), grayh - int(b[2])), (int(b[3]), grayh - int(b[4])), (0, 255, 0), 2)

                if showimages: cv2.imshow('Output Boxes Image', img)

                cv2.waitKey(0)

                #iterate through words

                for i in range(len(d['text'])):
                    file2.write('{0}: '.format(i) + d['text'][i] + '\n')
                    """
                    if d['text'][i] != '': #tesseract recognizes characters in current box
                        # expand box around each word
                        top = max(d['top'][i]-exppix,0)
                        bot = min(d['top'][i]+d['height'][i]+exppix,grayh)
                        left = max(d['left'][i]-exppix,0)
                        right = min(d['left'][i]+d['width'][i]+exppix,grayw)
                        # find individual characters in word and test
                        self.splitChars(gray[top:bot, left:right], showimages, d['left'][i], d['top'][i], d['width'][i], d['height'][i], d['conf'][i], d['text'][i])
                    """
                file2.close()



                # delete the temporary image
                os.remove(filename)

                cv2.waitKey(0)

    def splitChars(self, imggray, showimages, left, top, width, height, conf, text):
        # find character locations in current word, pass to Keras to test for
        # parameters:
            # imggray - image already cropped to current word

        h, w = imggray.shape

        #imggray = cv2.GaussianBlur(imggray, (5, 5), 0)
        ret,imggray = cv2.threshold(imggray, 200, 255, cv2.THRESH_BINARY)# | cv2.THRESH_OTSU)[1]
        #imggray = cv2.adaptiveThreshold(imggray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        ## Applied dilation
        #kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        #imggray = cv2.morphologyEx(imggray, cv2.MORPH_ERODE, kernel3)

        boxes = pytesseract.image_to_boxes(imggray , config='--psm 13')
        #dtemp = pytesseract.image_to_data(imggray, output_type=Output.DICT)
        for b in boxes.splitlines():
            b = b.split(' ')
            imggray = cv2.rectangle(imggray, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)



        if showimages: cv2.imshow('Output Character Boxes', imggray)

        cv2.waitKey(0)

        #https://medium.com/@quangnhatnguyenle/detect-and-recognize-vehicles-license-plate-with-machine-learning-and-python-part-2-plate-de644de9849f
        #cont, _ = cv2.findContours(imggray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #a = 1








"""psm tesseract options:
0 = Orientation and script detection (OSD) only.
1 = Automatic page segmentation with OSD.
2 = Automatic page segmentation, but no OSD, or OCR. (not implemented)
3 = Fully automatic page segmentation, but no OSD. (Default)
4 = Assume a single column of text of variable sizes.
5 = Assume a single uniform block of vertically aligned text.
6 = Assume a single uniform block of text.
7 = Treat the image as a single text line.
8 = Treat the image as a single word.
9 = Treat the image as a single word in a circle.
10 = Treat the image as a single character.
11 = Sparse text. Find as much text as possible in no particular order.
12 = Sparse text with OSD.
13 = Raw line. Treat the image as a single text line,
     bypassing hacks that are Tesseract-specific.
     
"""
