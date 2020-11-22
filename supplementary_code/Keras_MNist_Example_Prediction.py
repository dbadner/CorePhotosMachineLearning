# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np


# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    img = 1 - img #DTB Added invert colours
    return img


# load an image and predict the class
def run_example():

    # load model
    model = load_model('mnist_number_model.h5')

    for i in range(1, 6):
        filenm = 'input\\char{0}.jpg'.format(i)
        # load the image
        img = load_image(filenm)
        # predict the class
        digit = model.predict_classes(img)
        out_vec = model.predict(img) #returns a 1 x 10 output vector with probability of each digit
        p = np.amax(out_vec)
        str_out = "{} ({:.2f})".format(digit[0], p)
        print(str_out)


# entry point, run the example
run_example()
