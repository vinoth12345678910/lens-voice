# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset.csv")

print(df.head())
print(df.shape)


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

# Change "target" to your actual target column
X = df.drop("target", axis=1)
y = df["target"]


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

# Fill numerical missing values
X = X.fillna(X.median(numeric_only=True))


# ============================================================
# 5. CONVERT CATEGORICAL COLUMNS TO NUMBERS
# ============================================================

X = pd.get_dummies(X, drop_first=True)


# ============================================================
# 6. SPLIT DATASET
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 7. CREATE XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric="logloss"
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 10. CHECK ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# ============================================================
# 11. CREATE CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# ============================================================
# 12. VISUALIZE CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()





# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset.csv")

print(df.head())
print(df.shape)


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

# Change "target" to your actual target column
X = df.drop("target", axis=1)
y = df["target"]


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

X = X.fillna(X.median(numeric_only=True))


# ============================================================
# 5. CONVERT CATEGORICAL COLUMNS
# ============================================================

X = pd.get_dummies(X, drop_first=True)


# ============================================================
# 6. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 7. SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# 8. CREATE MODEL
# ============================================================

model = LogisticRegression(max_iter=1000)


# ============================================================
# 9. TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# 10. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 11. CHECK ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# ============================================================
# 12. CREATE CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# ============================================================
# 13. VISUALIZE CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()






# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from lightgbm import LGBMClassifier


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset.csv")

print(df.head())
print(df.shape)


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

# Change "target" to your actual target column
X = df.drop("target", axis=1)
y = df["target"]


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

X = X.fillna(X.median(numeric_only=True))


# ============================================================
# 5. CONVERT CATEGORICAL COLUMNS TO NUMBERS
# ============================================================

X = pd.get_dummies(X, drop_first=True)


# ============================================================
# 6. SPLIT DATASET
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 7. CREATE LIGHTGBM MODEL
# ============================================================

model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=-1,
    random_state=42,
    verbosity=-1
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 10. CHECK ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# ============================================================
# 11. CREATE CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# ============================================================
# 12. VISUALIZE CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix - LightGBM")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()






# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from catboost import CatBoostClassifier


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset.csv")

print(df.head())
print(df.shape)


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

# Change "target" to your actual target column
X = df.drop("target", axis=1)
y = df["target"]


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

X = X.fillna(X.median(numeric_only=True))


# ============================================================
# 5. CONVERT CATEGORICAL COLUMNS
# ============================================================

X = pd.get_dummies(X, drop_first=True)


# ============================================================
# 6. SPLIT DATASET
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 7. CREATE CATBOOST MODEL
# ============================================================

model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=False
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

# Convert CatBoost output to 1D
y_pred = y_pred.flatten()


# ============================================================
# 10. CHECK ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# ============================================================
# 11. CREATE CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# ============================================================
# 12. VISUALIZE CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix - CatBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()