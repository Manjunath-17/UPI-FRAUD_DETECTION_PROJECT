import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load preprocessed data
data = pd.read_csv(r"C:\Users\manjunath s khot\Documents\2KD22CS049\UPI-FRAUD_DETECTION_PROJECT\cleaned_upi_fraud_dataset.csv")

X = data.drop(columns=['fraud_risk'])
y = data['fraud_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=500),
    'Decision Tree': DecisionTreeClassifier(max_depth=20, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', C=1)
}

accuracies = {}

# Train and evaluate
for name, model in models.items():
    print()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    accuracies[name] = acc

    print(f"{name} Accuracy: {acc:.2f}")
    print(f"{name} Precision: {prec:.2f}")
    print(f"{name} Recall: {rec:.2f}")
    print(f"{name} F1-Score: {f1:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
    plt.title(f'Confusion Matrix - {name}')
    plt.show()

# Accuracy comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), hue=list(accuracies.keys()), legend=False, palette='viridis')
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.show()

# Save trained models
for name, model in models.items():
    joblib.dump(model, f"{name.lower().replace(' ', '_')}_model.pkl")

print("✅ Models trained and saved successfully!")
