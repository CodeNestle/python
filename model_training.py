import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load Data
DATA_DIR = "sign_data"
X = []
y = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".npy"):
        label = file.split("_")[0]
        landmarks = np.load(os.path.join(DATA_DIR, file))
        X.append(landmarks)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Test Accuracy
accuracy = clf.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the Model
joblib.dump(clf, "hand_sign_model.pkl")
print("Model saved as hand_sign_model.pkl")
