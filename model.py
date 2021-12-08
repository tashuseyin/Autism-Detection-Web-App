import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

data = pd.read_csv("dataset/data.csv")
print(data.head())
print(data.shape)
print(data.describe().T)
data.info()

# Gereksiz Öznitelikleri silme
data.drop(["Case_No", "Who completed the test"], axis=1, inplace=True)
print(data.columns)

# data öznitelik yeniden isimlendirme
data.rename(columns={"Class/ASD Traits ": "target"}, inplace=True)


# Categorical Variable
def bar_plot(variable, color):
    var = data[variable]
    varValue = var.value_counts()

    # visualize
    plt.figure(figsize=(9, 3))
    plt.bar(varValue.index, varValue, color=color)
    plt.xticks(varValue.index, varValue.index.values, rotation=45)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()

    print("{}:\n{}".format(variable, varValue))


category = ["Qchat-10-Score", "Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD", "target"]
color_list = ["red", "green", "blue", "orange", "yellow", "purple"]
for i in range(6):
    bar_plot(category[i], color_list[i])

# Numerical Variable
plt.figure(figsize=(9, 6))
plt.hist(data["Age_Mons"], bins=50)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Hist ile Age dağılımı")
plt.show()


# Outlier Detection
def detect_outliers(df, features):
    outlier_indices = []

    for c in features:
        Q1 = np.percentile(df[c], 25)
        Q3 = np.percentile(df[c], 75)
        IQR = Q3 - Q1

        outlier_step = IQR * 1.5
        outlier_list = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    return multiple_outliers


print(data.loc[detect_outliers(data, ["Age_Mons"])])

# Missing Value
data.isnull().sum()


def uniqueValue(variable):
    return data[variable].unique()


for i in data.columns:
    print(f"{i}: ", uniqueValue(i))

plt.figure(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True)
plt.show()

# Kategorik verileri dönüştürme
from sklearn.preprocessing import LabelEncoder

labelEncoder = LabelEncoder()
category = ["Qchat-10-Score", "Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD", "target"]
for i in category:
    data[i] = labelEncoder.fit_transform(data[i])

X = data.drop(["target"], axis=1)
y = data["target"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

from sklearn.metrics import accuracy_score, confusion_matrix

print("Test Accuracy Score: {}".format(accuracy_score(y_test, y_pred_test)))
print("Test Confusion matrix:\n {}".format(confusion_matrix(y_test, y_pred_test)))
print("Train Accuracy Score: {}".format(accuracy_score(y_train, y_pred_train)))
print("Train Confusion matrix:\n {}".format(confusion_matrix(y_train, y_pred_train)))

# Hyperparametre tuning
from sklearn.model_selection import GridSearchCV

params = {'n_neighbors': np.arange(0, 30)}
knn = KNeighborsClassifier()
knn_cv_model = GridSearchCV(knn, param_grid=params, cv=10)
knn_cv_model.fit(X_train, y_train)

print("En iyi parametre:", knn_cv_model.best_params_["n_neighbors"])

# hiperparametre tuning manuel
acc_score = []
for n in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc_score.append(accuracy_score(y_test, y_pred))
print(acc_score)
plt.plot(range(1, 30), acc_score)
plt.xlabel('n_neighbors')
plt.ylabel('Acuracy Scores')
plt.show()

# Tuned model
knn_tuned = KNeighborsClassifier(n_neighbors=25)
knn_tuned.fit(X_train, y_train)

y_pred_test = knn_tuned.predict(X_test)
y_pred_train = knn_tuned.predict(X_train)
print("Test Accuracy Score: {}".format(accuracy_score(y_test, y_pred_test)))
print("Test Confusion matrix:\n {}".format(confusion_matrix(y_test, y_pred_test)))
print("Train Accuracy Score: {}".format(accuracy_score(y_train, y_pred_train)))
print("Train Confusion matrix:\n {}".format(confusion_matrix(y_train, y_pred_train)))

import pickle

pickle.dump(knn_tuned, open("model.pkl", "wb"))

