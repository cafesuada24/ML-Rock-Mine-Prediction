# Built-in
import os

# Data manipulation
import numpy as np
import pandas as pd

# Data prep
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix

# Algorithms
from sklearn.linear_model import LogisticRegression



random_state = 2024
directory = os.getcwd()
file = r'\data\SONAR.csv'
with open(directory + file, 'r') as f:
    sonar_df = pd.read_csv(f, header = None)

X = sonar_df.iloc[:, 0:-1]
y = sonar_df.iloc[:, -1].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .1, stratify=y, random_state=random_state)


k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

penalty = 'l2'
C = 1.0
class_weight = 'balanced'
solver = 'liblinear'

log_reg = LogisticRegression(penalty=penalty, C=C, class_weight=class_weight, solver=solver)


training_scores = []
cv_scores = []
predictions = pd.Series(index=y_train.index)

model = log_reg
model.fit(X_train, y_train)
