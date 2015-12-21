__author__ = 'Eran Yogev and Orel Elimelech'

#MUST READ: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html, predict_proba VS decision_function


import numpy as np
import scipy as sc
from sklearn import metrics
from numpy import genfromtxt
import csv
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
# Import the classifiers
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from nltk.classify import SklearnClassifier
from sklearn.linear_model import SGDClassifier
import nltk.classify
from sklearn.svm import LinearSVC


'''
# Dataset size: 182997
# Records with length = 1: 128075
'''

print '****************'
print("Starting classifying application...")

# Read the data set
is_multi = False
if is_multi:
    dataset_file = 'dataset_all_threads_all_features_multi.csv'
else:
    dataset_file = 'dataset_all_threads_all_features.csv'
thread_length_dataset = genfromtxt(dataset_file, delimiter=',', skip_header=1, dtype=float)
with open(dataset_file,'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    headers = headers[1:-1]

dataset_size = thread_length_dataset.shape[0]
num_of_features = thread_length_dataset.shape[1]-2

#train_test_cutoff = 15000          #only for small experiments to reduce run-time
#test_validation_cutoff = 20000     #only for small experiments to reduce run-time

train_test_cutoff = 137247
test_validation_cutoff = 182997



#array[rows:columns]
# train_features = thread_length_dataset[0:train_test_cutoff, 1:-1]
# test_features = thread_length_dataset[train_test_cutoff:, 1:-1]
# train_labels = thread_length_dataset[0:train_test_cutoff, -1]
# test_labels = thread_length_dataset[train_test_cutoff:, -1]


train_features = thread_length_dataset[0:train_test_cutoff, 1:-1]
test_features = thread_length_dataset[train_test_cutoff:test_validation_cutoff, 1:-1]
train_labels = thread_length_dataset[0:train_test_cutoff, -1]
test_labels = thread_length_dataset[train_test_cutoff:test_validation_cutoff, -1]

print "train_features.shape:"+str(train_features.shape)
print "test_features.shape:"+str(test_features.shape)
print "train_labels.shape:"+str(train_labels.shape)
print "test_labels.shape:"+str(test_labels.shape)

if is_multi:
    print "Counting occurences."
    print "Count train labels:"+ str(Counter(train_labels))
    print "Count test labels:"+ str(Counter(test_labels))

print "Build the model"

''' Coose Classifier'''
#clf = tree.DecisionTreeClassifier() ; clf_name = "DecisionTree Classifier"
#clf = BaggingClassifier(n_estimators=15) ; clf_name = "Bagging Classifier"
#clf = BaggingClassifier(tree.DecisionTreeClassifier(class_weight = 'auto')) ; clf_name = "Bagging Classifier - Weighted"
#clf = RandomForestClassifier(n_estimators=15, class_weight='auto') ; clf_name = "RandomForest Classifier"
clf = AdaBoostClassifier() ; clf_name = "AdaBoost Classifier"
#clf = AdaBoostClassifier(tree.DecisionTreeClassifier(class_weight = 'auto')) ; clf_name = "AdaBoost Classifier - Weighted"
#clf = GradientBoostingClassifier(n_estimators=200,learning_rate=0.1,min_samples_leaf=25) ; clf_name = "GradientBoosting Classifier"
#clf = GaussianNB() ; clf_name = "GaussianNB Classifier"
#clf = SGDClassifier(loss="hinge", penalty="l2"); clf_name = 'SGDClassifier'
#clf = LinearSVC() ; clf_name = "LinearSVC"


''' Fit Data'''
clf = clf.fit(train_features, train_labels); clf_params = "None"
#clf = clf.fit(train_features, train_labels, sample_weight=np.array([2.1 if i == 1 else 1 for i in train_labels])) ; clf_params = "sample_weight=np.array([2.1 if i == 1 else 1 for i in train_labels])"



print "Calculate the error"
predictions = clf.predict(test_features)
try:
    labels_predicted_prob = clf.predict_proba(test_features)
    prob_calc = True
except:
    prob_calc = False



try:
    feature_importance = clf.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    print "Rank the features by their importance"
    importance = True
except:
    importance = False

if not is_multi:
    print "Compute ROC curve and ROC area"
    y_score_ok = False
    try:# This is correct; http://scikit-learn.org/stable/auto_examples/plot_roc.html
        y_score = clf.decision_function(test_features)
        y_score_ok = True
    except:# This is correct: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
        if(prob_calc):
            y_score = labels_predicted_prob[:,1]
            y_score_ok = True
    if(y_score_ok is True):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr[0], tpr[0], _ = roc_curve(test_labels[:], y_score[:])
        roc_auc[0] = auc(fpr[0], tpr[0])


print '****************'
# Print general information
print "Classifier: " + clf_name
print "Parameters: " + clf_params
if not is_multi:
    if(y_score_ok):
        print("AUC: " + str(metrics.roc_auc_score(test_labels,y_score[:])))# TODO: Change labels_predicted_prob[:,1] to y_score
    v_precision, v_recall, _ = precision_recall_curve(test_labels[:], y_score[:])
    average_precision = average_precision_score(test_labels[:], y_score[:],average="micro")
    #print "Precision:{0}".format(v_precision)
    #print "Recall:{0}".format(v_recall)
    #print "Thresholds:{0}".format(v_thresholds)
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
if importance:
    # Print the ranking of the features
    print("Features Importance:")
    for id in reversed(sorted_idx):
        if (feature_importance[id]>0):
            print(headers[id]+": "+str(feature_importance[id]))

if not is_multi:

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
    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(v_recall, v_precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision))
    plt.legend(loc="lower left")
    plt.show()


print("Done.")

