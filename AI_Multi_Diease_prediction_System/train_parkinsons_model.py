import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("dataset/parkinsons.csv")

# Remove patient name
data = data.drop("name", axis=1)

# Features and target
X = data.drop("status", axis=1)
y = data["status"]
print(X.columns)
print(len(X.columns))


# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("models/parkinsons_model.pkl", "wb"))

print("Parkinson Model Trained Successfully!")