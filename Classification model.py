#Convert preprocessed text of CoViNAR dataset into vectors/embeddings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


X.shape #X contains the embeddings
X = X.reshape(X.shape[0], -1)
y=np.array(df['Num_label']) #y have the labels of corresponding tweet

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



lr_model = LogisticRegression(max_iter=1000, penalty= 'l2',solver='lbfgs',class_weight='balanced',
                           C = 4,n_jobs=-1,random_state=0)
svc_model = SVC(kernel = 'poly', C=10, class_weight='balanced',random_state=40)
RF_model = RandomForestClassifier(n_estimators=1000, random_state=0, criterion='entropy',max_features='sqrt', n_jobs=-1,
                               min_samples_split=2)

# Use 10-fold cross validation
skf = StratifiedKFold(n_splits=10)

scores = cross_val_score(svc_model, X_train, y_train, cv=skf)
print("Stratified Cross-validation scores:", scores)
print("Average cross-validation score: ", scores.mean())

svc_model.fit(X_train, y_train)

# Evaluate on test set
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = svc_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy: ", accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred, average = 'weighted')
print("Test Precision: ", precision)

# Calculate recall
recall = recall_score(y_test, y_pred, average = 'weighted')
print("Test Recall: ", recall)

# Calculate F1-score
f1 = f1_score(y_test, y_pred, average = 'weighted')



# Assuming you have the true labels (y_true) and predicted labels (y_pred) from your classification model

cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(3, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


#Plot Precision-Recall graph
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np


# # Create an instance of OneVsRestClassifier with Logistic Regression
clf = OneVsRestClassifier(SVC(kernel = 'poly', C=10, class_weight='balanced',random_state=40, probability=True))

# # Fit the model
clf.fit(X_train, y_train)

# Get the underlying binary estimators
estimators = clf.estimators_

# Compute precision-recall curve for each class
n_classes = 3
precision, recall, thresholds = dict(), dict(), dict()

for i in range(n_classes):
    y_score = estimators[i].predict_proba(X_test)[:, 1]
    precision[i], recall[i], thresholds[i] = precision_recall_curve(y_test == i, y_score)

# Plot the precision-recall curves
plt.figure(figsize=(8, 6))
lines = []
labels = []
for i in range(n_classes):
    line, = plt.plot(recall[i], precision[i], lw=2)
    lines.append(line)
    labels.append(f'Class {i}')

plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve for Multiclass Classification', fontsize=16)
plt.legend(lines, labels, loc='lower left', fontsize=12)
plt.show()

