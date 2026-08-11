import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Load dataset
df = pd.read_csv('data/churn.csv')

# Convert TotalCharges to numeric and clean
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Drop ID column — no predictive value
df.drop('customerID', axis=1, inplace=True, errors='ignore')

# Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# EDA plots
sns.countplot(x='Churn', data=df); plt.title("Churn Count"); plt.show()
sns.countplot(x='Contract', hue='Churn', data=df); plt.title("Contract vs Churn"); plt.show()
sns.boxplot(x='Churn', y='MonthlyCharges', data=df); plt.title("Monthly Charges vs Churn"); plt.show()

# Encode categoricals and split
df = pd.get_dummies(df, drop_first=True)
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Train and evaluate
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
