import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load Dataset
data = pd.read_csv("dataset/heart.csv")

# Features and Target
X = data.drop("target", axis=1)
y = data["target"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model
pickle.dump(model, open("models/heart_model.pkl", "wb"))

print("Heart Model Trained Successfully!")