import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            precision_recall_fscore_support, roc_curve, auc,
                            classification_report, confusion_matrix)

import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('creditcard.csv')
df.head()
df.info()
df.isnull().sum()
df.describe()
df['Class'].value_counts()
positive_class = df[df['Class'] == 1].shape[0]/df['Class'].shape[0]
negative_class = 1-positive_class
print(f'{100*positive_class:.3f}%')
print(f'{100*negative_class:.3f}%')
pd.set_option('display.max_columns', None)
df.groupby('Class').mean()
plt.figure(figsize=(15,20))
for i,col in enumerate(df.columns[:-1], 1):
    plt.subplot(10,3,i)
    plt.hist(df[col], bins=35)
    plt.title(f'{col} Distribution')

plt.tight_layout()
plt.show()
pd.set_option('display.max_columns', None)
df.groupby('Class').mean()
df.groupby('Class').max()
sns.boxplot(x=df['Class'], y=df['Amount'])
plt.title('Transaction Amount by each Class')
plt.show()

x1 = df[df['Class']==0]['Amount'].mean()
x2 = df[df['Class']==1]['Amount'].mean()
std1 = df[df['Class']==0]['Amount'].std()
std2 = df[df['Class']==1]['Amount'].std()
n = df[df['Class']==0].shape[0]
m = df[df['Class']==1].shape[0]

z = (x1 - x2) / np.sqrt((std1**2 / n) + (std2**2 / m))
print(f'z statistic: {z:.3f}')

lower_critical_value = -1.96
upper_critical_value = 1.96
if z < lower_critical_value or z > upper_critical_value:
    print("There is a difference between the means of the two populations.")
else:
    print("There is no significant difference between the means of the two populations.")

majority_class_df = df[df['Class'] == 0]
minority_class_df = df[df['Class'] == 1]

random_subset_majority_class = majority_class_df.sample(n=10000, random_state=42)
df = pd.concat([random_subset_majority_class, minority_class_df])
df.shape
df.drop('Time', axis=1)
X = df.drop('Class', axis=1).values
y = df['Class'].values
n_splits = 5
stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

model = LogisticRegression()

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for train_index, test_index in stratified_kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, pos_label=1))
    recall_scores.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

print("Mean Accuracy:", sum(accuracy_scores) / n_splits)
print("Mean Precision:", sum(precision_scores) / n_splits)
print("Mean Recall:", sum(recall_scores) / n_splits)

log_model =LogisticRegression()

n_splits=5
kfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

mean_recall_values = []
mean_precision_values = []
mean_f1_values = []

for threshold in thresholds:
    recall_values = []
    precision_values = []
    f1_values = []

    for train_index, test_index in kfolds.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        log_model.fit(X_train, y_train)
        probabilities = log_model.predict_proba(X_test)[:, 1]

        predictions = (probabilities > threshold).astype(int)
        precision, recall, _, _ = precision_recall_fscore_support(y_test, predictions, average='binary')

        recall_values.append(recall)
        precision_values.append(precision)
        f1 = 2 * (precision * recall) / (precision + recall)
        f1_values.append(f1)

    mean_recall = np.mean(recall_values)
    mean_precision = np.mean(precision_values)
    mean_f1 = np.mean(f1_values)

    mean_recall_values.append(mean_recall)
    mean_precision_values.append(mean_precision)
    mean_f1_values.append(mean_f1)

    print(f"Threshold: {threshold:.2f}, Mean Precision: {mean_precision:.2f}, Mean Recall: {mean_recall:.2f}, Mean F1 Score: {mean_f1:.2f}")

print(f"Overall Mean Precision: {np.mean(mean_precision_values):.3f}, Overall Mean Recall: {np.mean(mean_recall_values):.3f}, Overall Mean F1 Score: {np.mean(mean_f1_values):.2f}")

best_threshold_index = np.argmax(mean_recall_values)
best_threshold = thresholds[best_threshold_index]

print(f"Best Threshold: {best_threshold:.2f} with Mean Recall: {np.max(mean_recall_values):.4f} and Mean Precision {mean_precision_values[best_threshold_index]:.4f}")
log_reg_recall = np.max(mean_recall_values)
