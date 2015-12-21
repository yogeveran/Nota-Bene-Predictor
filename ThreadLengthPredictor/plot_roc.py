__author__ = 'Eran Yogev'

import csv
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn import metrics

print '****************'
print("Starting classifying application...")
print '****************'

print("Loading data...")

# Read the data set
dataset_file = 'dataset_all_threads_all_features_multi.csv'
thread_length_dataset = genfromtxt(dataset_file, delimiter=',', skip_header=1, dtype=float)
with open(dataset_file,'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    headers = headers[1:-1]

dataset_size = thread_length_dataset.shape[0]
num_of_features = thread_length_dataset.shape[1]-2

print "dataset_size:"+str(dataset_size)
print "num_of_features:"+str(num_of_features)


# array[rows:columns]
all_features = thread_length_dataset[:, 2:-1] # All rows, ommit first two:"location_id","ctime" and last "thread_length" collums.
all_labels = thread_length_dataset[:, -1] # All rows,only last "thread_length" collums.

print "all_features.shape:"+str(all_features.shape)
print "all_labels.shape:"+str(all_labels.shape)

y = all_labels
X = all_features

print '****************'
print("Done loading data...")
print '****************'
# quit()


print "Binarize the output"
y = label_binarize(y, classes=[1, 2, 3, 4])
n_classes = y.shape[1]


print "shuffle and split training and test sets"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

print "Train Model: Learn to predict each class against the other"
#classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
#                                 random_state=0))#random_state=random_state


classifier = OneVsRestClassifier(AdaBoostClassifier())
#classifier = OneVsRestClassifier(AdaBoostClassifier(tree.DecisionTreeClassifier(class_weight = 'auto')))
classifier = classifier.fit(X_train, y_train)
y_score = classifier.decision_function(X_test)
predictions = classifier.predict(X_test)
try:
    labels_predicted_prob = classifier.predict_proba(X_test)
except:
    labels_predicted_prob = None
    print 'Unable to calculate labels_predicted_prob'
# y_score = classifier.fit(X_train, y_train).decision_function(X_test)

print "Compute ROC curve and ROC area for each class"
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print '*******************'
    print "Label {0}\n".format(i)
    print ("classification report:")
    print str(metrics.classification_report(y_test[:, i], predictions[:, i]))
    if not (labels_predicted_prob is None):
        # AUC
        try:
            auc_str = str(metrics.roc_auc_score(y_test[:, i], labels_predicted_prob[:, i]))
            print '*******************'
            print 'AUC:'
            print auc_str
        except:
            print "Unable to print AUC."

    # Confusion Matrix
    try:
        confusion = str(metrics.confusion_matrix(y_test[:, i], predictions[:, i]))
        print '*******************'
        print 'Confusion Matrix:'
        print confusion
    except:
        print "Unable to print Confusion Matrix."
    # Accuracy Score
    try:
        accuracy = str(metrics.accuracy_score(y_test[:, i], predictions[:, i]))
        print '*******************'
        print 'Accuracy Score:'
        print accuracy
    except:
        print "Unable to print Accuracy Score."
print "Compute micro-average ROC curve and ROC area"
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

'''
print "Plot of a ROC curve for a specific class"
plt.figure()
plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

'''

print "Plot ROC curve"
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
