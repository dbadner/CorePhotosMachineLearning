import imutils
import csv


def resize_image(img, max_w, max_h):
    # resizes the cv2 image based on the maximum width and maximum height, returns the resized image
    if img.ndim == 2:  # black and white
        (tH, tW) = img.shape
    else:  # colour
        (tH, tW, tmp) = img.shape
    # if the width is greater than the height (usually will be), resize along the width dimension
    if tW / tH > max_w / max_h:  # width is the defining dimension along which to resize
        hCheck = tH / (tW / max_w)
        if hCheck > 1.00001:  # check not < 1, will throw error
            img = imutils.resize(img, width=max_w)
        else:
            img = imutils.resize(img, width=max_w, height=1)
    # otherwise, resize along the height
    else:
        w_check = tW / (tH / max_h)
        if w_check > 1.00001:  # check not < 1, will throw error
            img = imutils.resize(img, height=max_h)
        else:
            img = imutils.resize(img, height=max_h, width=1)
    return img


def output_to_csv(fname, data_list):
    # function outputs dataset to CSV
    writemode = 'w'
    with open(fname, writemode) as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        for dataLine in data_list:
            outputlist = []
            for a in dataLine:
                outputlist.append(str(a))
            filewriter.writerow(outputlist)
