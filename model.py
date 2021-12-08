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
data.drop(["Case_No", "Who completed the test", "Qchat-10-Score"], axis=1, inplace=True)
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


category = ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD", "target"]
color_list = ["green", "blue", "orange", "yellow", "purple"]
for i in range(5):
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
category = ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD", "target"]
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
knn_tuned = KNeighborsClassifier(n_neighbors=11)
knn_tuned.fit(X_train, y_train)

y_pred_test = knn_tuned.predict(X_test)
y_pred_train = knn_tuned.predict(X_train)
print("Test Accuracy Score: {}".format(accuracy_score(y_test, y_pred_test)))
print("Test Confusion matrix:\n {}".format(confusion_matrix(y_test, y_pred_test)))
print("Train Accuracy Score: {}".format(accuracy_score(y_train, y_pred_train)))
print("Train Confusion matrix:\n {}".format(confusion_matrix(y_train, y_pred_train)))

import pickle

pickle.dump(knn_tuned, open("knn_model.pkl", "wb"))

# Random Forests
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
print(rf_model)

# ilkel test hatamız
y_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))

# model tuning
rf_params = {"max_depth": [2, 3, 5, 8, 10],
             "max_features": [2, 5, 8],
             "n_estimators": [10, 500, 1000],
             "min_samples_split": [2, 5, 10]}

rf_cv_model = GridSearchCV(rf_model,
                           rf_params,
                           cv=10, n_jobs=-1, verbose=2)

rf_cv_model.fit(X_train, y_train)

print("En iyi parametreler: {}".format(str(rf_cv_model.best_params_)))

# Tuned edilmiş final model
model_tuned = RandomForestClassifier(max_depth=2, max_features=2, min_samples_split=10, n_estimators=500)
model_tuned.fit(X_train, y_train)

y_pred_final_test = model_tuned.predict(X_test)
print("Test Score: {}".format(accuracy_score(y_test, y_pred_final_test)))

y_pred_final_train = model_tuned.predict(X_train)
print("Train Score: {}".format(accuracy_score(y_train, y_pred_final_train)))

pickle.dump(model_tuned, open("random_forest_model.pkl", "wb"))
