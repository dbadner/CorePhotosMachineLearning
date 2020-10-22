import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import Functions as fn

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

# then scale the data
sc = StandardScaler()
data = sc.fit_transform(data)

#print the data to test scaling
#for i in range(data.shape[1]):
#	print('>%d, min=%.3f, max=%.3f, avg=%.3f' % (i, data[:, i].min(), data[:, i].max(), data[:, i].mean()))

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3,
                                                    random_state=3)  # 70% training and 30% test



#SVM Polynomial model ###USED
#Create a svm Classifier
clf = svm.SVC(kernel='poly', degree=4, class_weight='balanced', gamma='scale', random_state=15, probability=True)

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_trainpred = clf.predict(X_train)
y_pred = clf.predict(X_test)
y_trainpred_prob = clf.predict_proba(X_train)
y_pred_prob = clf.predict_proba(X_test)

print("TRAIN")

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_train, y_trainpred))

# Model Precision: # correctly predicted positives / # predicted as positive
print("Precision:",metrics.precision_score(y_train, y_trainpred))

# Model Recall: # correctly predicted positives / # actual positives
print("Recall:",metrics.recall_score(y_train, y_trainpred))

# F1 Score F1 = 2 * (precision * recall) / (precision + recall)
print("F1 Score:",metrics.f1_score(y_train, y_trainpred))

print("TEST")

# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Model Precision: # correctly predicted positives / # predicted as positive
print("Precision:", metrics.precision_score(y_test, y_pred))

# Model Recall: # correctly predicted positives / # actual positives
print("Recall:", metrics.recall_score(y_test, y_pred))

# F1 Score F1 = 2 * (precision * recall) / (precision + recall)
print("F1 Score:", metrics.f1_score(y_test, y_pred))

score = clf.score(X_test, y_test)
cm = metrics.confusion_matrix(y_test, y_pred)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {:.2f}'.format(score)
plt.title(all_sample_title, size = 15);
plt.show()

#from: https://machinelearningmastery.com/how-to-save-and-load-models-and-data-preparation-in-scikit-learn-for-later-use/
from pickle import dump
# save the model
dump(clf, open('word_number_model_prob.pkl', 'wb'))
# save the scaler
dump(sc, open('word_number_scaler_prob.pkl', 'wb'))

#inverse transform the dataset and export it for viewing
X_test_inv = sc.inverse_transform(X_test)
X_train_inv = sc.inverse_transform(X_train)

X_inv = np.vstack((X_test_inv, X_train_inv))

Y_in = np.hstack((y_test, y_train))
Y_out = np.hstack((y_pred, y_trainpred))
Y_probout = np.vstack((y_pred_prob, y_trainpred_prob))

outArr = np.hstack((np.stack((Y_in, Y_out), axis=1), Y_probout, X_inv))

outList = outArr.tolist()
fn.OutputToCSV("TrainedCSVOut.csv", outList)
xxx = 1