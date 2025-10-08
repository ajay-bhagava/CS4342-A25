import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(y_test, y_pred):
    # Get class labels for confusion matrix
    class_labels = sorted(y_test.unique().tolist())
    # Model Evaluation with Confusion Matrix
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    conf_matrix = confusion_matrix(y_test, y_pred, labels=class_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=class_labels, yticklabels=class_labels)

    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.show()


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

# Feature Scaling
scaler = StandardScaler().set_output(transform="pandas")
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Random Forest Classifier")
evaluate(y_test, y_pred)
print("---------")

# LGBM Classsifier
lgbm = LGBMClassifier(nestimators=100, random_state=42, verbosity=-1)
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
print("LGBM Classifier")
evaluate(y_test, y_pred)
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

plt.figure(figsize=(8,6))
plt.title(f'Top {top_n} Feature Importances')
plt.bar(range(top_n), importances[top_indices], align='center')
plt.xticks(range(top_n), feature_names[top_indices], rotation=45)
plt.tight_layout()
plt.show()

