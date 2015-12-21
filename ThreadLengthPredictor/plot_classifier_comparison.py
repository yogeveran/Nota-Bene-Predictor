import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from numpy import genfromtxt
import csv


import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve




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
X = thread_length_dataset[:, 1:-1]
y = thread_length_dataset[:, -1]
X_train = thread_length_dataset[0:train_test_cutoff, 1:-1]
X_test = thread_length_dataset[train_test_cutoff:, 1:-1]
y_train = thread_length_dataset[0:train_test_cutoff, -1]
y_test = thread_length_dataset[train_test_cutoff:, -1]


# Create classifiers
lr = LogisticRegression()
gnb = GaussianNB()
svc = LinearSVC(C=1.0)
rfc = RandomForestClassifier(n_estimators=100)


###############################################################################
# Plot calibration plots

plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for clf, name in [(lr, 'Logistic'),
                  (gnb, 'Naive Bayes'),
                  (svc, 'Support Vector Classification'),
                  (rfc, 'Random Forest')]:
    clf.fit(X_train, y_train)
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(X_test)
        prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (name, ))

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
plt.show()