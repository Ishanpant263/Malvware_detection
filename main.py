import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file = 'TUANDROMD.csv'
try:
    dp = pd.read_csv(file)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{file}' was not found. Please check the file path.")
    exit()

print("Dataset shape:", dp.shape)
print("Dataset data types:")
print(dp.dtypes)

dp = dp.dropna(subset=['Label'])
print(f"Dataset shape after dropping missing labels: {dp.shape}")

print("Number of malware samples:", dp[dp['Label'] == 'malware'].shape[0])
print("Number of goodware samples:", dp[dp['Label'] == 'goodware'].shape[0])

dp['Label'] = dp['Label'].map({'goodware': 0, 'malware': 1})
target = dp['Label']
dp = dp.drop(['Label'], axis=1)

scaler = StandardScaler()
dp_scaled = scaler.fit_transform(dp)

x_train, x_test, y_train, y_test = train_test_split(dp_scaled, target, test_size=0.3, random_state=42)
print("Training set shape:", x_train.shape)
print("Testing set shape:", x_test.shape)

clf = RandomForestClassifier(max_depth=3, random_state=0)
model_rf = clf.fit(x_train, y_train)

train_pred_rf = model_rf.predict(x_train)
train_accuracy_rf = accuracy_score(y_train, train_pred_rf)
print("Random Forest - Training data accuracy:", train_accuracy_rf)

test_pred_rf = model_rf.predict(x_test)
test_accuracy_rf = accuracy_score(y_test, test_pred_rf)
print("Random Forest - Testing data accuracy:", test_accuracy_rf)

cm = confusion_matrix(y_test, test_pred_rf, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Goodware', 'Malware'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

feature_importances = model_rf.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=dp.columns)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
