import numpy as np
from pickle import load

# read in csv
datasetPath = 'depth_train_dataset.csv'
dataList = []
labelList = []
for i, row in enumerate(open(datasetPath)):
    if i == 0: continue  # skip first row, header rown
    # parse the label and image from the row
    row = row.split(",")
    # print(row)
    label = int(row[0])
    datarow = np.array([x for x in row[1:6]], dtype="float32")

    # print(datarow)

    dataList.append(datarow)
    labelList.append(label)
data = np.array(dataList, dtype="float32")
labels = np.array(labelList, dtype="int")

# load the model
model = load(open('word_number_model_prob.pkl', 'rb'))
# load the scaler
scaler = load(open('word_number_scaler_prob.pkl', 'rb'))

# then scale the data
data = scaler.transform(data)

#print the data to test scaling
for i in range(data.shape[1]):
	print('>%d, min=%.3f, max=%.3f, avg=%.3f' % (i, data[:, i].min(), data[:, i].max(), data[:, i].mean()))