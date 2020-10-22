import imutils
import csv

def ResizeImage(img, maxW, maxH):

    ''
    # resizes the cv2 image based on the maximum width and maximum height, returns the resized image
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


def OutputToCSV(fname, dataList):
    # function outputs dataset to CSV
    # check if file exists:
    #exists = os.path.exists(fname)
    writemode = 'w'
    #if exists: writemode = 'a'
    with open(fname, writemode) as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        for dataLine in dataList:
            outputlist = []
            for a in dataLine:
                outputlist.append(str(a))
            filewriter.writerow(outputlist)