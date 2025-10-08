import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(font_scale=1.25)

def evaluate(y_test, y_pred, model_name):
    # Get class labels for confusion matrix
    class_labels = sorted(y_test.unique().tolist())
    # Model Evaluation with Confusion Matrix
    # Get metrics 
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    precision = precision_score(y_test, y_pred, average='weighted')
    print(f'Precision: {precision * 100:.2f}%')

    recall = recall_score(y_test, y_pred, average='weighted')
    print(f'Recall: {recall * 100:.2f}%')

    # Get confusion matrix and plot
    conf_matrix = confusion_matrix(y_test, y_pred, labels=class_labels)

    plt.figure(model_name, figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=class_labels, yticklabels=class_labels, annot_kws={"size": 20})

    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()


# df = pd.read_csv(r'Stress_Dataset.csv')
df = pd.read_csv(r'StressLevelDataset.csv')

# Features
X = df.drop(["stress_level"], axis=1)

# Data Labels
y = df["stress_level"].astype("category")

# Get unique class labels from your dataset
class_labels = sorted(y.unique().tolist())

# Partitioning Dataset into Test and Validation Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72)

# Feature Scaling, scale to normal distribution mean=0, std.dev=1
scaler = StandardScaler().set_output(transform="pandas")
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Random Forest Classifier")
evaluate(y_test, y_pred, "Random Forest")
print("---------")

# LGBM Classsifier
lgbm = LGBMClassifier(nestimators=100, random_state=42, verbosity=-1)
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
print("LGBM Classifier")
evaluate(y_test, y_pred, "LightGBM")
print("---------")

# Feature Importance
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]

feature_names = df.columns[:-1]
print("\nFeature Importances:")
for idx in indices:
    print(f"{feature_names[idx]}: {importances[idx]:.4f}")

# Plot feature importances  
top_n = 8
top_indices = indices[:top_n]

plt.figure("FeatureImportance", figsize=(8,6))
plt.title(f'Top {top_n} Feature Importances')
plt.bar(range(top_n), importances[top_indices], align='center')
plt.xticks(range(top_n), feature_names[top_indices], rotation=45)
plt.tight_layout()
plt.show()


