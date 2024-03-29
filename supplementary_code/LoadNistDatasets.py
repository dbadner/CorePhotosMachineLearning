# code from: https://www.pyimagesearch.com/2020/08/17/ocr-with-keras-tensorflow-and-deep-learning/

# import the necessary packages
from tensorflow.keras.datasets import mnist
import numpy as np
import h5py


class LoadNistDatasets:

    def load_az_dataset_csv(self, datasetPath):
        # initialize the list of data and labels
        data = []
        labels = []
        # loop over the rows of the A-Z handwritten digit dataset
        for row in open(datasetPath):
            # parse the label and image from the row
            row = row.split(",")
            label = int(row[0])
            image = np.array([int(x) for x in row[1:]], dtype="uint8")
            # images are represented as single channel (grayscale) images
            # that are 28x28=784 pixels -- we need to take this flattened
            # 784-d list of numbers and repshape them into a 28x28 matrix
            image = image.reshape((28, 28))
            # update the list of data and labels
            data.append(image)
            labels.append(label)

        # convert the data and labels to NumPy arrays
        data = np.array(data, dtype="float32")
        labels = np.array(labels, dtype="int")
        # return a 2-tuple of the A-Z data and labels
        return (data, labels)

    def load_az_dataset(self, datasetPath):
        #load the az dataset saved in hdf5 file
        # retrieve datasets
        f2 = h5py.File(datasetPath, 'r')
        azDataRead = f2['azData'][:]
        azLabelsRead = f2['azLabels'][:]
        f2.close()
        return (azDataRead, azLabelsRead)

    def load_mnist_dataset(self):
        # load the MNIST dataset and stack the training data and testing
        # data together (we'll create our own training and testing splits
        # later in the project)
        ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
        data = np.vstack([trainData, testData])
        labels = np.hstack([trainLabels, testLabels])
        # return a 2-tuple of the MNIST data and labels
        return (data, labels)
