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
