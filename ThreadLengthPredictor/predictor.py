__author__ = 'Eran Yogev and  Orel Elimelech'

import numpy as np
import scipy as sc
from sklearn.datasets import load_iris
from sklearn import metrics
from numpy import genfromtxt
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Import the classifiers
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

'''
# Dataset size: 182997
# Records with length = 1: 128075
'''

'''
iris = load_iris()
X, y = iris.data, iris.target
print X.shape
print X
print y
X_new = SelectKBest(f_classif, k=2).fit_transform(X, y)
print X_new.shape
print X_new
'''

#train_test_cutoff = 15000          #only for small experiments to reduce run-time
#test_validation_cutoff = 20000     #only for small experiments to reduce run-time
# train_test_cutoff = 91498
# test_validation_cutoff = 137247

train_test_cutoff = 137247
test_validation_cutoff = 182997

print '****************'
print("Starting classifying application...")

# Read the data set
dataset_file = 'dataset_all_threads_all_features.csv'
thread_length_dataset = genfromtxt(dataset_file, delimiter=',', skip_header=1, dtype=float)
with open(dataset_file,'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    headers = headers[1:-1]
print("Done loading dataset...")
'''
np.seterr(divide='ignore', invalid='ignore')
new_features_set = SelectKBest(f_classif, k=44).fit_transform(thread_length_dataset[:,1:-1],thread_length_dataset[:,-1])
'''


dataset_size = thread_length_dataset.shape[0]
num_of_features = thread_length_dataset.shape[1]

#array[rows:columns]
train_features = thread_length_dataset[0:train_test_cutoff, 1:-1]
test_features = thread_length_dataset[train_test_cutoff:test_validation_cutoff, 1:-1]
train_labels = thread_length_dataset[0:train_test_cutoff, -1]
test_labels = thread_length_dataset[train_test_cutoff:test_validation_cutoff, -1]
print("Building model...")
# Build the model
#clf = tree.DecisionTreeClassifier() ; clf_name = "DecisionTree Classifier"
#clf = BaggingClassifier(n_estimators=15) ; clf_name = "Bagging Classifier"
#clf = RandomForestClassifier(n_estimators=15, class_weight='auto') ; clf_name = "RandomForest Classifier"
clf = AdaBoostClassifier() ; clf_name = "AdaBoost Classifier"
#clf = GradientBoostingClassifier(n_estimators=200,learning_rate=0.1,min_samples_leaf=25) ; clf_name = "GradientBoosting Classifier"
#clf = GaussianNB() ; clf_name = "GaussianNB Classifier"
#clf = SVC() ; clf_name = "SVC Classifier"

clf = clf.fit(train_features, train_labels, sample_weight=np.array([2.1 if i == 1 else 1 for i in train_labels])) ; clf_params = "sample_weight=np.array([2.1 if i == 1 else 1 for i in train_labels])"
#clf = clf.fit(train_features, train_labels)
print("Done building model...")


# Calculate the error
predictions = clf.predict(test_features)
labels_predicted_prob = clf.predict_proba(test_features)

# Rank the features by their importance
feature_importance = clf.feature_importances_
sorted_idx = np.argsort(feature_importance)

# Compute ROC curve and ROC area
y_score = clf.decision_function(test_features)
fpr = dict()
tpr = dict()
roc_auc = dict()
'''
#For one label
fpr[0], tpr[0], _ = roc_curve(test_labels[:], y_score[:])
roc_auc[0] = auc(fpr[0], tpr[0])

#For multiLabelfpr --- Need to binarize
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
'''

print '****************'
# Print general information
print "Classifier: " + clf_name
print "Parameters: " + clf_params
print("AUC: " + str(metrics.roc_auc_score(test_labels,labels_predicted_prob[:,1])))
print ("Accuracy Score: " + str(metrics.accuracy_score(test_labels, predictions)))
print "Data set size: " + str(dataset_size)
print "Number of features: " + str(num_of_features)
print "Train records: " + str(1) + "-" + str(train_test_cutoff)
print "Test records: " + str(train_test_cutoff+1) + "-" + str(test_validation_cutoff)
# print "Validation records: " + str(test_validation_cutoff + 1) + "-" + str(dataset_size)
print '****************'
# Print the confusion matrix
print 'Confusion Matrix:'
print metrics.confusion_matrix(test_labels,predictions)
print '****************'
# Print recall and precision
print 'Classification Report:'
print metrics.classification_report(test_labels,predictions)
print '****************'
print "Features: "
print headers
print '****************'
# Print the ranking of the features"imports
print("Features Importance:")
for id in reversed(sorted_idx):
    if (feature_importance[id]>0):
        print(headers[id]+": "+str(feature_importance[id]))
'''
print '****************'
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
print '****************'
'''
print("Done.")

