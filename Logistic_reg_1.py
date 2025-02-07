# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# %%
ad_data = pd.read_csv(r'C:\Real junkies\sem 8\Datasets\advertising.csv')

# %%
ad_data.head()

# %%
ad_data.info()

# %%
ad_data.describe()

# %%
sns.set_style('whitegrid')
ad_data['Age'].hist(bins = 30)
plt.xlabel('Age')

# %%
sns.jointplot(x = 'Age', y = 'Area Income', data = ad_data)

# %%
sns.jointplot(x = 'Age', y = 'Daily Time Spent on Site', data = ad_data, color = 'red', kind = 'kde');

# %%
sns.jointplot (x = 'Daily Time Spent on Site', y = 'Daily Internet Usage', data = ad_data, color = 'green')

# %%
sns.pairplot(ad_data, hue = 'Clicked on Ad', palette = 'bwr')

# %%
x = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# %%
logmodel = LogisticRegression(max_iter= 500) ## here 100 iterations were not enough to converge
logmodel.fit(x_train, y_train)

# %%
predictions = logmodel.predict(x_test)

# %%
from sklearn.metrics import classification_report  ## to create a classification report of the model

# %%
print(classification_report(y_test, predictions))

# %% [markdown]
# Precision : out of all predicted positives , how many were actually correct 
# 
# TP / (TP + FP)
# 
# TP is True Positive (Actual value was 1 and predicted is also 1)
# 
# FP is False positive (Actual value was 0 but predicted 1)
# 
# FN is False negative (Actual value was 1 but predicted 0)
# 
# TN is True negative (actual value was 0 and predicted also 0)
# 
# 
# recall: Out of all actual positives / (TP and FN), how many did the model correctly predict
# 
# TP / (TP + FN)
# 
# 
# F1-score is a "balance" between precision and recall (harmonic mean)
# 
# Support is the total no. of instances count of 0s + 1s
# 
# So, high TP and TN = Good model performance 
# 

# %% [markdown]
# # Testing the model

# %%
new_data = pd.DataFrame([{
    'Daily Time Spent on Site': 80,
    'Age': 60,
    'Area Income': 55000,
    'Daily Internet Usage': 200,
    'Male': 0
}])

# %%
import sklearn
import numpy as np
print(sklearn.__version__)  # Check if scikit-learn is installed
print(np.__version__)  # Check if numpy is installed
print(StandardScaler)

import sklearn

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform (x_train) ## here we needed scalar to get initialised and also have some trained dataset scaled as well
x_test_scaled = scaler.transform (x_test)

# %%
new_data_scaled = scaler.transform (new_data)

# %% [markdown]
# ### The new data scaled
# 
# [[-3.25625154e-01 -6.85319482e-01 -4.54719034e-03  4.21437876e-01
#    3.90845932e+01]]

# %%
model_prediction = logmodel.predict (new_data_scaled)
print("Prediction:", "Yes" if model_prediction[0] == 1 else "No")

# %% [markdown]
# Therefore, it predicted by the given data that it clicked on ad.

# %%
from sklearn.metrics import roc_curve, auc
y_pred_proba = logmodel.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve (y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# %%
print ("Area Under the Curve (AUC):", roc_auc)

# %%
import matplotlib.pyplot as plt 
plt.figure (figsize = (8,6))
plt.plot (fpr, tpr, color = 'darkorange', lw = 2, label = f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot  ([0,1], [0,1], color = 'navy', lw = 2, linestyle = '--')
plt.xlim ([0.0, 1.0])
plt.ylim ([0.0, 1.05])
plt.xlabel ('False Positive Rate')
plt.ylabel ('True Positive Rate')
plt.title ('Receiver Operating Characteristic (ROC) Curve')
plt.legend (loc = "lower right")
plt.show()

# %%



