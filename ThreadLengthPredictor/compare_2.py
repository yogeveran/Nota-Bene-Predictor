__author__ = 'eran'

import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import csv
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score, classification_report)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn import metrics
import inspect
from sklearn.metrics import roc_auc_score

# Read the data set
is_multi = False

def read_data():
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

    train_test_cutoff = 137247
    test_validation_cutoff = 182997

    #array[rows:columns]
    X = thread_length_dataset[:, 1:-1]
    y = thread_length_dataset[:, -1]
    X_train = thread_length_dataset[0:train_test_cutoff, 1:-1]
    X_test = thread_length_dataset[train_test_cutoff:, 1:-1]
    y_train = thread_length_dataset[0:train_test_cutoff, -1]
    y_test = thread_length_dataset[train_test_cutoff:, -1]

    return X, y, X_train, X_test, y_train, y_test, headers

X, y, X_train, X_test, y_train, y_test, headers = read_data()

def  plot_calibration_curve(fig_index, classifiers):
    """Plot calibration curve for classifiers"""

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))
        print "\tClassification Report:{0}\n".format(classification_report(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()

def plot_precision_recall_curve(fig_index, classifiers):
    """Plot Precision Recall curve for classifiers"""

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=3)

    for clf, name in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))
        print "\tClassification Report:{0}\n".format(classification_report(y_test, y_pred))

        v_precision, v_recall, _ = precision_recall_curve(y_test, prob_pos)
        average_precision = average_precision_score(y_test, prob_pos)

        ax1.plot(v_recall, v_precision, "s-",
                 label="%s (%1.3f)" % (name, average_precision))

    ax1.set_ylabel("Precision")
    ax1.set_xlabel("Recall")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Precision Recall Curve')

    plt.tight_layout()

def weighted_plot_precision_recall_curve(fig_index, classifiers, print_feat):
    """Plot Precision Recall curve for classifiers"""

    weight_function = np.array([2.1 if i == 1 else 1 for i in y_train])
    weight_function_str = "np.array([2.1 if i == 1 else 1 for i in y_train])"
    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=3)

    for clf, name in classifiers:
        clf.fit(X_train, y_train)

        if hasattr(clf, "feature_importances_"):
            feature_importance = clf.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            importance = True
        else:  # use decision function
            importance = False

        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        v_precision, v_recall, _ = precision_recall_curve(y_test, prob_pos)
        average_precision = average_precision_score(y_test, prob_pos)#This score corresponds to the area under the precision-recall curve.
        roc_auc = roc_auc_score(y_test, prob_pos)

        print '*********************************'
        print("%s:" % name)
        print '\tPR_AUC:\n\t{0}\n'.format(average_precision)
        print '\tROC_AUC:\n\t{0}\n'.format(roc_auc)
        print '\tConfusion Matrix:\n\t{0}\n'.format(metrics.confusion_matrix(y_test, y_pred))
        print "\tClassification Report:\n{0}\n".format(classification_report(y_test, y_pred))
        if importance and print_feat:
            # Print the ranking of the features
            print("\tFeatures Importance:\n")
            for id in reversed(sorted_idx):
                if (feature_importance[id]>0):
                    print(headers[id]+": "+str(feature_importance[id]))
        print '*********************************'

        ax1.plot(v_recall, v_precision, "s-",
                 label="%s (%1.3f)" % (name, average_precision))

    for clf, name in classifiers:
        if 'sample_weight' in inspect.getargspec(clf.fit)[0]:
            clf.fit(X_train, y_train, sample_weight=weight_function)
            name += '- Weighted: ' + weight_function_str
        else:
            continue


        if hasattr(clf, "feature_importances_"):
            feature_importance = clf.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            importance = True
        else:  # use decision function
            importance = False

        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        v_precision, v_recall, _ = precision_recall_curve(y_test, prob_pos)
        average_precision = average_precision_score(y_test, prob_pos)
        roc_auc = roc_auc_score(y_test, prob_pos)
        print '*********************************'
        print("%s:" % name)
        print '\tPR_AUC:\n\t{0}\n'.format(average_precision)
        print '\tROC_AUC:\n\t{0}\n'.format(roc_auc)

        print '\tConfusion Matrix:\n\t{0}\n'.format(metrics.confusion_matrix(y_test, y_pred))
        print "\tClassification Report:\n{0}\n".format(classification_report(y_test, y_pred))
        if importance and print_feat:
            # Print the ranking of the features
            print("\tFeatures Importance:\n")
            for id in reversed(sorted_idx):
                if (feature_importance[id]>0):
                    print(headers[id]+": "+str(feature_importance[id]))
        print '*********************************'

        ax1.plot(v_recall, v_precision, "s-",
                 label="%s (%1.3f)" % (name, average_precision))

    ax1.set_ylabel("Precision")
    ax1.set_xlabel("Recall")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Precision Recall Curve')

    plt.tight_layout()


'''
classifiers = [(GaussianNB(), "Naive Bayes"),
               (AdaBoostClassifier(), "AdaBoost"),
               (RandomForestClassifier(n_estimators=100), "RandomForest"),
               (BaggingClassifier(n_estimators=100), "Bagging"),
               (LogisticRegression(C=1., solver='lbfgs'), "LogisticRegression"),
               (CalibratedClassifierCV(AdaBoostClassifier(), cv=2, method='isotonic'),"AdaBoost + isotonic calibration"),
               (CalibratedClassifierCV(AdaBoostClassifier(), cv=2, method='sigmoid'),"AdaBoost + sigmoid calibration")
               ]
'''
classifiers = [(AdaBoostClassifier(), "AdaBoost"),
               (RandomForestClassifier(n_estimators=80), "RandomForest"),
               (GaussianNB(), "Naive Bayes")
               ]



# Plot calibration curve for classifiers
weighted_plot_precision_recall_curve(1, classifiers,False)


plt.show()