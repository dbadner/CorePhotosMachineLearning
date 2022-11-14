# modified from: https://www.pyimagesearch.com/2020/08/24/ocr-handwriting-recognition-with-opencv-keras-and-tensorflow/
# identify characters within an image
# import the necessary packages
# handle warnings
import warnings
warnings.filterwarnings('ignore')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
from pickle import load
import numpy as np
import imutils
import cv2
import h5py
import csv
import Functions as fn
import re


class WBImage:
    # declare class variables
    InputDir: str  # image input directory
    Num_AZ_Model: object  # input neural network model for predicting most likely a-z and 0 - 9 character combined
    Num_Model: object  # input neural network model for predicting most likely 0 - 9 character combined
    LabelNames = [l for l in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    charList: list  # list of Character class objects
    wordList: list  # list of Word class objects
    keyWordList: list  # list of keyWord class objects
    image: object  # image in colour, scaled with aspect ratio to max 2000 x 2000
    imageOutAnno: object  # image annotated for output to FormUI, showing identified keywords, identified numbers, and
    # characters
    gray: object  # image in grayscale, scaled with aspect ratio to max 2000 x 2000
    # variables for displaying on output form:
    depthFrom: str  # depth from found for the current image, in string format
    depthTo: str  # depth to found for the current image, in string format
    depthFromP: float  # corresponding probability [0:1]
    depthToP: float  # corresponding probability [0:1]
    wetDry: str  # string, "WET" if "WET" found, or "DRY" if "DRY" found
    wetDryP: float  # corresponding probability [0:1]
    DevelopMode: bool  # if True, will show stepwise images through code more interactively, needed for training

    # default false for general use of program

    # CharList: list #list of Character class objects found in image by neural network
    # WordList: list #list of Word class objects comprising 2 or more characters nearby
    # ProbNumList: list  # list corresponding to wordList indices with probability of number
    # WordCharList: list  # list of character arrays in words corresponding to wordList
    # KeyWordList: list  # list of char lists defining keywords

    def __init__(self, inputdir: str):
        self.DevelopMode = False  # SET TO TRUE FOR DEBUGGING
        self.TrainingMode = False  # SET TRUE FOR BUILDING TRAINING SET, ALONG WITH DEVELOP MODE ABOVE
        self.InputDir = inputdir
        self.Num_AZ_Model = load_model('number_az_model.h5')
        self.Num_Model = load_model('mnist_number_model.h5')
        self.charList = []
        self.wordList = []
        self.build_key_word_list()
        self.depthFrom = ""
        self.depthTo = ""
        self.wetDry = ""
        self.wetDryP = 0.0


    def build_key_word_list(self):
        self.keyWordList = []
        self.keyWordList.append(KeyWord(['F', 'R', 'O', 'M'], 1, self.LabelNames))
        self.keyWordList.append(KeyWord(['T', 'O'], 1, self.LabelNames))  # IGNORING DEPTH TO AS OF 28-OCT, TOO
        # ERRONEOUS
        self.keyWordList.append(KeyWord(['D', 'E', 'P', 'T', 'H'], 2, self.LabelNames))  # allow for two depths to be
        # found
        self.keyWordList.append(KeyWord(['D', 'R', 'Y'], 1, self.LabelNames))
        self.keyWordList.append(KeyWord(['W', 'E', 'T'], 1, self.LabelNames))

    def preprocess(self):
        # convert image to grayscale, and blur it to reduce noise
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.gray = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # Applied dilation
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.gray = cv2.morphologyEx(self.gray, cv2.MORPH_ERODE, kernel3)

    def find_chars_words(self):
        # perform edge detection, find contours in the edge map, and sort the
        # resulting contours from left-to-right, find words

        edged = cv2.Canny(self.gray, 30, 150)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # cv2.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)
        cnts = sort_contours(cnts, method="left-to-right")[0]
        # initialize the list of contour bounding boxes and associated
        # characters that we'll be ocr'ing
        # loop over the contours and populate if they pass the criteria
        self.process_chars(cnts)
        return

    def process_chars(self, cnt):
        # function takes in image and contour and filters the characters to those within words,
        # and those with appropriate sizes, and adjusts white space

        for i in range(len(cnt)):
            c = cnt[i]
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            # filter out bounding boxes, ensuring they are neither too small
            # nor too large
            if (5 <= w <= 375) and (5 <= h <= 300) and w / h < 22 and h / w < 22:

                # extract the character and threshold it to make the character
                # appear as *white* (foreground) on a *black* background, then
                # grab the width and height of the thresholded image
                roi = self.gray[y:y + h, x:x + w]
                thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

                thresh = fn.resize_image(thresh, 22, 22)  # resize the image

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
                # prepare the padded image for classification via our
                # handwriting ocr model
                padded = padded.astype("float32") / 255.0
                padded = np.expand_dims(padded, axis=-1)
                # check if small character, unlikely text (could be punctuation):
                small_char_filt = False
                if h < 25:
                    small_char_filt = True
                self.charList.append(Character(padded, (x, y, w, h), False, small_char_filt,
                                               False))  # update our list of characters that will be ocr'd

        # check each character to make sure not overlapping with an another character, discard if so
        remove_list = self.check_overlap()
        for i in remove_list:
            self.charList[i].OvlFilt = True

        # now loop through chars and perform checks, assign to words
        for i, char in enumerate(self.charList):
            if not char.OvlFilt:
                fnd_word = False
                (x, y, w, h) = char.Dims  # read in character dimensions
                for j, word in enumerate(self.wordList):  # loop through existing word list
                    prev_char = self.charList[
                        word.charList[len(word.charList) - 1]]  # read in previous character in word
                    fnd_word = self.char_checks(i, char, prev_char, j, word)
                    if fnd_word:
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
                if not fnd_word:  # start a new word
                    new_word = Word()
                    new_word.dims = char.Dims
                    new_word.charList = [i]
                    self.wordList.append(new_word)
                    char.InWord = True

        # now loop back through words and reassess first character vs second character spacing (skipped before)
        nnn = 0
        yyy = 1
        for i, word in enumerate(self.wordList):
            if len(word.charList) > 1:
                char1 = self.charList[word.charList[0]]
                x1 = char1.Dims[0] + char1.Dims[2]
                char2 = self.charList[word.charList[1]]
                x2 = char2.Dims[0]
                if not self.check_mean_spacing(x1, x2, word):  # remove first character if true, spacing too large
                    self.charList[word.charList[0]].InWord = False
                    word.charList.remove(word.charList[0])  # remove the first character

                    self.update_word_vals(word)

        # final loop to throw out words that only have one character
        for i in range(len(self.wordList) - 1, -1, -1):
            words = self.wordList[i]
            if len(words.charList) < 2:
                self.charList[words.charList[0]].InWord = False  # change back to false, no longer in a word
                del self.wordList[i]

    # return charList, wordList

    def update_word_vals(self, word):
        # function iterates through word parameters and updates values

        (xW, yW, wW, hW) = self.charList[word.charList[0]].Dims  # initialize word values to first character values
        wW = self.charList[word.charList[len(word.charList) - 1]].Dims[2] + \
             self.charList[word.charList[len(word.charList) - 1]].Dims[0] - xW

        avgSpac = 0
        avgH = 0
        avgW = 0
        for n, charInd in enumerate(word.charList):
            (x, y, w, h) = self.charList[charInd].Dims
            yWN = min(y, yW)
            hWN = max(y + h, yW + hW) - yWN
            yW = yWN
            hW = hWN
            avgH += h
            avgW += w
            if n > 0:  # skip the first iteration, no previous character
                (xP, yP, wP, hP) = self.charList[word.charList[n - 1]].Dims  # previous character dims
                avgSpac += x - (xP + wP)

        avgH /= len(word.charList)
        avgW /= len(word.charList)
        if len(word.charList) > 1:
            avgSpac /= len(word.charList) - 1

        word.avgCharSpac = avgSpac
        word.avgCharH = avgH
        word.avgCharW = avgW

        word.dims = (xW, yW, wW, hW)

    def char_checks(self, i, char, prev_char, j, word):
        # function runs a series of checks to check whether character 'char' is in word 'word', returns true is so, or false if not
        (x, y, w, h) = char.Dims  # read in character dimensions
        (xW, yW, wW, hW) = word.dims  # read in word dimensions
        (xC, yC, wC, hC) = prev_char.Dims  # read in last character dimensions in word
        # compare to determine whether character is part of current word
        xDif = x - (xW + wW)  # check x
        if xDif >= hW / 1.2:  # compare to word height to check if close enough to word to be included (i.e. whitespace between)
            return False
        # check y-overlap against previous character in word (instead of fill word
        if y > (yC + hC) or (y + h) < yC:  # need to also check amount of y-overlap
            return False
        ovl = (min(y + h, yC + hC) - max(y, yC)) / (max(y + h, yC + hC) - min(y, yC))  # percentage overlap
        if ovl <= 0.3:  # set 30% overlap threshold
            return False
        ht_ratio = h / hW
        if ht_ratio > 2.5 or ht_ratio < 0.3:  # thresholds for height ratios
            return False
        if not (len(word.charList) <= 3 or (len(word.charList) > 2 and self.check_mean_spacing(xW + wW, x, word))):
            return False
        # final check - look for a change in average spacing between characters in a word
        else:
            return True

    @staticmethod
    def check_mean_spacing(x1, x2, word):
        # Function performs check of spacing between characters relative to mean spacing of characters in word
        # Returns True if spacing criteria is okay, false is failed (i.e. too large spacing)
        # variables: x2 - left of second character, x1 - right of first character, word = current word, cutoff is
        # cutoff value
        cutoff = 0.4
        ret = (max(x2 - x1, 0) - word.avgCharSpac) / word.avgCharH < cutoff
        return ret

    def check_overlap(self):
        # check each character to make sure not overlapping with an another character
        remove_list = []
        for i, charI in enumerate(self.charList):
            (x, y, w, h) = charI.Dims  # read in character dimensions
            discard: bool = False
            for j, charJ in enumerate(self.charList):
                if j == i:
                    continue
                (xC, yC, wC, hC) = charJ.Dims  # read in character dimensions
                if w * h <= wC * hC:  # only discard the smaller of the two
                    if ((xC < x < xC + wC) or (xC < x + w < xC + wC)) and \
                            ((yC < y < yC + hC) or (yC < y + h < yC + hC)):
                        # there is overlap, determine how much as proportion of smaller item
                        a_ovl = (min(x + w, xC + wC) - max(x, xC)) * (min(y + h, yC + hC) - max(y, yC))
                        perc_ovl = a_ovl / (w * h)
                        if perc_ovl > 0.6:  # overlap > 60%
                            remove_list.append(i)
                            break
        return remove_list

    def run_model(self, image_file, output_anno_dir):
        # run the model to predict characters
        # image_file is the filename

        # function level variables
        # define the list of label names
        img_anno = self.image.copy()  # make a local copy of coloured image

        dry_ind = 3
        wet_ind = 4
        keyword_prob_min = 0.5  # if < 50% then ignore keyword

        # extract the bounding box locations and padded characters
        boxes = [b.Dims for b in self.charList]
        image_data = np.array([c.Data for c in self.charList], dtype="float32")
        # ocr the characters using our handwriting recognition model
        preds = self.Num_AZ_Model.predict(image_data)  # predict most likely num and char
        preds_num = self.Num_Model.predict(image_data)  # predict most likely num

        # loop over the predictions and bounding box locations together
        for n, (pred, (x, y, w, h)) in enumerate(zip(preds, boxes)):
            # find the index of the label with the largest corresponding
            # probability, then extract the probability and label
            i = np.argmax(pred)
            prob = pred[i]
            label = self.LabelNames[i]
            # draw the prediction on the image
            if self.DevelopMode:  # only show cv2 image if in develop mode
                cv2.rectangle(img_anno, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if self.charList[n].OvlFilt == False and self.charList[n].SmallFilt == False:
                    # only add text to the image if filter out flag is false
                    cv2.putText(img_anno, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
        if self.DevelopMode:  # only show cv2 image if in develop mode
            # loop over words, draw rectangle around
            for wInd, words in enumerate(self.wordList):
                (x, y, w, h) = words.dims
                cv2.rectangle(img_anno, (x, y), (x + w, y + h), (255, 0, 255), 2)  # rectangle around word

            # show the image
            image_s = fn.resize_image(img_anno, 800, 800)
            cv2.imshow("Image", image_s)
            cv2.waitKey(0)

        # loop over words again
        for wInd, words in enumerate(self.wordList):
            prob_num = 0  # probability that word contains a number
            word_chars = []  # list of character indices in word
            # loop over characters in word and determine probability of #,
            for i in words.charList:  # for each ith character
                prob = preds[i]
                for j in range(10):  # loop through probability of number characters
                    prob_num += prob[j]
                word_chars.append(self.LabelNames[np.argmax(preds[i])])

            prob_num /= len(words.charList)
            words.probNum = prob_num
            words.wordCharList = word_chars

            # check probability of being a keyword
            for k in self.keyWordList:
                key_word_ind = k.CharsInd
                prob_key_word = 0
                if len(words.charList) == len(key_word_ind):  # same length, so check probability
                    for i in range(len(key_word_ind)):  # for each ith character
                        ind = words.charList[i]
                        prob_key_word += preds[ind][key_word_ind[i]]  # find probability that ith characters match
                    prob_key_word /= len(key_word_ind)
                    # now compare to list, and replace if the new highest likelihool of word is found
                    for n in range(len(k.MaxProb)):
                        if prob_key_word > k.MaxProb[n]:  # new highest likelihood found
                            k.MaxProb[n] = prob_key_word
                            k.MaxProbWordInd[n] = wInd  # corresponding word index
                            break
        # show image of keyWords picked out, and print probabilities correct
        print("-----")
        print("KEYWORD SEARCH SUMMARY:")
        self.imageOutAnno = self.image.copy()
        for ii, k in enumerate(self.keyWordList):
            for n in range(len(k.MaxProb)):
                temp: str = ""
                for c in k.Chars:
                    temp = temp + c
                temp = temp + ": P={:.1f}%".format(k.MaxProb[n] * 100)
                print(temp)
                execute = True
                if k.MaxProb[n] < keyword_prob_min:  # keyword not found with sufficient probability (40%)
                    execute = False
                if "".join(k.Chars) == "TO":  # skip if keyword is "TO" - ignoring this keyword as of 28-oct
                    execute = False
                # hardcode for DRY vs WET
                if ii == dry_ind and self.keyWordList[dry_ind].MaxProb[n] < self.keyWordList[wet_ind].MaxProb[n]:
                    # compare probability of dry vs wet, only show the higher probability
                    execute = False
                elif ii == wet_ind and self.keyWordList[wet_ind].MaxProb[n] < self.keyWordList[dry_ind].MaxProb[n]:
                    execute = False
                if execute:
                    (x, y, w, h) = self.wordList[k.MaxProbWordInd[n]].dims
                    cv2.rectangle(self.imageOutAnno, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    for i in range(len(k.Chars)):  # characters in keyword
                        label = k.Chars[i]
                        charInd = self.wordList[k.MaxProbWordInd[n]].charList[i]
                        (xC, yC, wC, hC) = boxes[charInd]
                        cv2.putText(self.imageOutAnno, label, (xC - 10, yC - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                    (255, 0, 0), 2)
                        # also draw box
                        cv2.rectangle(self.imageOutAnno, (xC, yC), (xC + wC, yC + hC), (0, 0, 255), 2)
                    # store wet vs dry in class variable, if that keyword
                    if ii == wet_ind or ii == dry_ind:
                        self.wetDry = "".join(self.keyWordList[ii].Chars)
        if self.DevelopMode:  # only show cv2 image if in develop mode
            image_s = fn.resize_image(self.imageOutAnno, 800, 800)
            cv2.imshow("Keywords Image", image_s)
            cv2.waitKey(0)

        inp_bool = False
        if self.TrainingMode:  # only write cv2 image if in training mode
            # For training mode: Training Set by saving photos to folders
            valid = False
            inp: str
            while not valid:
                inp = input("Add to training set? (y/n): ")
                if inp == "": inp = "Y"
                inp = inp.upper()
                if inp == "Y" or inp == "N":
                    valid = True
            if inp == "Y":  # write image to training set folder
                image_path = self.InputDir + '/' + image_file
                image_temp = cv2.imread(image_path)
                train_dir = self.InputDir + '/trainset'
                if not os.path.exists(train_dir): os.makedirs(train_dir)
                image_path_out = train_dir + "/" + image_file
                cv2.imwrite(image_path_out, image_temp)
                # build and scale feature matrix for words in current image, store in data_list and label_list
                inp_bool = True

        (tH, tW) = self.gray.shape
        data_list, label_list, punct_id_list = self.build_feature_matrix(inp_bool, tH, tW, self.imageOutAnno,
                                                                         keyword_prob_min)
        if self.TrainingMode and inp_bool:  # write to training set csv if trainingmode enabled and user specified 'y'
            self.save_update_training_set_csv('depth_train_dataset.csv', image_file, data_list, label_list)

        y_result = self.run_word_num_svm_model(
            data_list)  # output from 1 to 0 representing likelihood of being a number
        y_max_ind = [-1, -1]  # word index corresponding to highest and second highest y_result
        y_max = [0, 0]  # highest and second highest y_result
        for ind, (word, y) in enumerate(zip(self.wordList, y_result)):
            # code to output word characters and corresponding number likelihood
            if self.DevelopMode:
                output = "".join(word.wordCharList)
                output += ": {:.2f}".format(y[1])
                print(output)

            # check if most or second most likely, store if so
            if y[1] > y_max[0]:
                y_max[1] = y_max[0]
                y_max_ind[1] = y_max_ind[0]
                y_max[0] = y[1]
                y_max_ind[0] = ind
            elif y[1] > y_max[1]:
                y_max[1] = y[1]
                y_max_ind[1] = ind
        word_num_str = []  # list of str
        # output results to image
        for n, p in zip(y_max_ind, y_max):  # loop through 2 number words found...
            curr_word = ""
            if p > 0.15:  # only output if prob of word being number > 35%
                (x, y, w, h) = self.wordList[n].dims
                cv2.rectangle(self.imageOutAnno, (x, y), (x + w, y + h), (255, 0, 255), 2)
                # check for punctuation, add to image if it exists
                xP = yP = wP = hP = -1  # initialize punctuation coordinates to null val
                punct = False
                if punct_id_list[n] > -1:  # -1 is null val for no punctuation
                    (xP, yP, wP, hP) = self.charList[punct_id_list[n]].Dims
                    cv2.putText(self.imageOutAnno, ".", (xP - 10, yP - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (255, 0, 0), 2)
                    punct = True
                for c in self.wordList[n].charList:  # number characters,
                    # output box to image around character
                    (xC, yC, wC, hC) = self.charList[c].Dims
                    cv2.rectangle(self.imageOutAnno, (xC, yC), (xC + wC, yC + hC), (0, 0, 255), 2)
                    # pull labels from numeric neural network
                    i = np.argmax(preds_num[c])
                    prob = preds_num[i]
                    label = self.LabelNames[i]
                    cv2.putText(self.imageOutAnno, label, (xC - 10, yC - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (255, 0, 0), 2)
                    if punct and (xP + wP / 2) < (xC + wC / 2):
                        # there is punctuation, and if it fits before current character
                        curr_word += "."
                        punct = False
                    curr_word += label
            word_num_str.append(curr_word)
        # regardless of probability, assign number words to from / to respectively, store in class variables
        if self.wordList[y_max_ind[0]].dims[0] < self.wordList[y_max_ind[1]].dims[0]:
            # compare x values, depth from comes first
            self.depthFrom = word_num_str[0]
            self.depthFromP = y_max[0]
            self.depthTo = word_num_str[1]
            self.depthToP = y_max[1]
        else:
            self.depthFrom = word_num_str[1]
            self.depthFromP = y_max[1]
            self.depthTo = word_num_str[0]
            self.depthToP = y_max[0]

        print("-----")
        print("PREDICTION SUMMARY:")
        print("Depth From: " + self.depthFrom + " (P={:.0f}%)".format(self.depthFromP * 100))
        print("Depth To: " + self.depthTo + " (P={:.0f}%)".format(self.depthToP * 100))
        print("Wet / Dry: " + self.wetDry)

        if self.DevelopMode:  # only show cv2 image if in develop mode
            image_s = fn.resize_image(self.imageOutAnno, 800, 800)
            cv2.imshow("Final Image", image_s)
            cv2.waitKey(0)

        # lastly, save the image in the Output_Anno folder
        out_file_name: str = output_anno_dir + '/' + re.search(r"(.*)\.", image_file).group(0)[:-1]
        out_file_name += "_WB_Cropped_Anno.png"
        cv2.imwrite(out_file_name, self.imageOutAnno)

        return out_file_name

    def build_feature_matrix(self, labeldata: bool, tH, tW, image, keyword_prob_min):
        # built feature matrix
        # parameters:
        # labeldata: True if interactive user labelling data
        # features:
        # x_dist - x distance from nearest keyword, scaled to [-1,1] by bounds of whiteboard image
        # y_dist - y distance from nearest keyword, scaled to [-1,1] by bounds of whiteboard image
        # p_numb - average likelihood that characters in word are numeric [0,1]
        # punct - whether or not the word contains a '.' punctuation character, [0=False,1=True]
        # num_chars - #number of characters scaled
        # height - average height of the word
        featnames = ["x_dist", "y_dist", "prob_numb", "punct", "num_chars", "height"]
        data = []  # np.zeros(len(wordList), n_feat)
        labels = []  # 1 for depth value word, 0 for not. only relevant if labeldata = true, otherwise blank
        punct_id_list = []  # character ID of punction for each word, -1 if no punctuation

        # find maximum avgCharHt
        max_ht: float = 0.0
        for word in self.wordList:
            ht = word.avgCharH
            if ht > max_ht:
                max_ht = ht

        # obtain feature vector for each word
        for n, word in enumerate(self.wordList):
            x_dist, y_dist = self.find_closest_keyword(word, tH, tW, keyword_prob_min)
            p_numb = word.probNum
            punct, punct_id = self.find_punctuation(word)
            # punct is true or false, and if true, return charID for punctuation
            punct_id_list.append(punct_id)
            num_chars: float = self.find_num_chars_scaled(word)
            height = self.find_rel_height(word, max_ht)

            datarow = np.array([x_dist, y_dist, p_numb, punct, num_chars, height], dtype="float32")
            data.append(datarow)

            if labeldata:  # interactive user data labelling
                # first, print vector to screen:
                output = ("WORD #{}: ".format(n))
                output += "".join(word.wordCharList)
                print(output)
                for i, name in enumerate(featnames):
                    output = name + ": {:.2f}".format(datarow[i])
                    print(output)

                img = image.copy()
                (x, y, w, h) = word.dims
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 176, 240), 2)
                image_s = fn.resize_image(img, 800, 800)
                cv2.imshow("Keywords Image", image_s)
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

        return data, labels, punct_id_list

    @staticmethod
    def find_rel_height(word, max_ht):
        # function returns the avg height of the word divided by the avg height of the tallest word
        return word.avgCharH / max_ht

    @staticmethod
    def find_num_chars_scaled(word):
        # function returns number of words from word, scaled such that 3 - 6 characters are treated the same
        # optimized for numbers with two decimal places
        num_chars = len(word.charList)
        if num_chars > 3:
            num_chars -= min(3, num_chars - 3)
        return num_chars

    def find_closest_keyword(self, word, tH, tW, keyword_prob_min):
        # finds and returns x and y distance to closest keyword
        # searching keywords index 0 - 2
        (x, y, w, h) = word.dims
        xc = x + w / 2
        yc = y + h / 2
        xMinDif: float = tW  # minimum x and y difference between word center and keyword center, initialize to
        # image dimensions
        yMinDif: float = tH
        hMinDif: float = tW ** 2 + tH ** 2  # minimum hypoteneuse
        for k, keyWord in enumerate(self.keyWordList):
            if "".join(keyWord.Chars) == "TO": continue  # skip if "TO" keyword, no longer using
            if k >= 3: break  # exit loop if past 'depth' keyword, hardcoded
            for n, p in zip(keyWord.MaxProbWordInd, keyWord.MaxProb):
                if p < keyword_prob_min: continue  # ignore keyword if < min cutoff, currently 40%
                (xK, yK, wK, hK) = self.wordList[n].dims
                xcK = xK + wK / 2
                ycK = yK + hK / 2
                xdif = xc - xcK
                ydif = yc - ycK
                hdif = xdif ** 2 + ydif ** 2
                if abs(hdif) < abs(hMinDif):
                    hMinDif = hdif  # assign if closer
                    xMinDif = xdif
                    yMinDif = ydif
        # scale
        xMinDif /= tW
        yMinDif /= tH
        return xMinDif, yMinDif

    def find_punctuation(self, word):
        # function looks for punctuation sized character within word and returns true if found, false if not
        (xW, yW, wW, hW) = word.dims
        fnd_punct: bool = False
        punct_id: int = -1
        for ind, char in enumerate(self.charList):
            if char.InWord or char.SmallFilt == False:
                # character is not in a word, and has been flagged as a small character, possible punctuation
                continue
            (x, y, w, h) = char.Dims
            # check within lower quarter of word, expanded downward by 1/4 of word height
            if x < xW + (wW * 0.1) or x > xW + wW - (wW * 0.1):  # search middle 80% of word for punctuation
                continue  # not within x
            if y < yW + hW / 2 or y > yW + hW + hW / 4:
                continue  # not within reasonable y
            fnd_punct = True
            punct_id = ind
            break
        return float(fnd_punct), punct_id

    @staticmethod
    def save_update_training_set_csv(fname, imagefile, data_list, label_list):
        # function checks if an existing training set exists, and builds a new one if not
        # structure of CSV:
        # row 0 = labels
        # rows 1 - 6 = data
        # row 7 = image file name
        # check if file exists:
        exists = os.path.exists(fname)
        writemode = 'w'
        if exists: writemode = 'a'
        with open(fname, writemode) as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')
            if not exists:
                filewriter.writerow(["labels", "x_dist", "y_dist", "prob_numb", "punct", "num_chars", "height",
                                     "filename"])  # header row first
            for dataLine, labelLine in zip(data_list, label_list):
                outputlist = []
                outputlist.append(str(labelLine))
                for a in dataLine:
                    outputlist.append(str(a))
                outputlist.append(imagefile)  # image file name
                filewriter.writerow(outputlist)

    @staticmethod
    def run_word_num_svm_model(data_list):
        # function runs the SVM on the word dataset to classify if word is a number (1) or not (0)

        # load the model
        model = load(open('word_number_model_prob.pkl', 'rb'))
        # load the scaler
        scaler = load(open('word_number_scaler_prob.pkl', 'rb'))

        # scale the data using saved scaler
        data = np.array(data_list, dtype="float32")
        data = scaler.transform(data)

        yval_prob = model.predict_proba(data)

        return yval_prob

    @staticmethod
    def save_update_training_set(fname, imagefile, data_list, label_list):
        # FUNCTION NOT WORKING, not currently used
        # function checks if an existing training set exists, and builds a new one if not
        # check if file exists
        exists = os.path.exists(fname)
        data_all: list  # = []
        labalAll: list  # = []
        f: h5py.File
        if exists:
            f = h5py.File(fname, 'r+')
            try:
                data_read = f['data'][:]
                label_read = f['labels'][:]
                data_all = list(data_read)
                label_all = list(label_read)
            except:
                xxx = 1
        else:  # create new file
            f = h5py.File(fname, 'w')

        data_all2 = np.array(data_list, dtype="float32")
        label_all2 = np.array(label_list, dtype="int")
        f.create_dataset('data', data=data_all2)
        f.create_dataset('labels', data=label_all2)
        f.close()


class Word:
    dims: tuple  # = (0, 0, 0, 0)  #= np.zeros(4, dtype=int) #x,y,w,h
    charList = []  # list of character indices
    avgCharSpac: float = 0  # initialize average character spacing to -1
    avgCharW: float = 0
    avgCharH: float = 0
    probNum: float = 0
    wordCharList: list = []  # most likely word characters based on NN model output


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
        self.CharsInd = self.assign_chars_ind(chars, labelNames)

    @staticmethod
    def assign_chars_ind(chars: list, label_names: list):
        key_word_ind = []
        for char in chars:
            for i in range(len(label_names)):
                l = label_names[i]
                if l == char:
                    key_word_ind.append(i)
                    break
        return key_word_ind
