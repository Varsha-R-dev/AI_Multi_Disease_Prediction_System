import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load Dataset
data = pd.read_csv("dataset/kidney.csv")

# Replace ? with NaN
data.replace("?", pd.NA, inplace=True)

# Remove extra spaces from text
data = data.apply(lambda col: col.astype(str).str.strip())

# Convert text values to numbers
data.replace({
    'normal': 0,
    'abnormal': 1,
    'present': 1,
    'notpresent': 0,
    'yes': 1,
    'no': 0,
    'good': 1,
    'poor': 0,
    'ckd': 1,
    'notckd': 0,
    'Male': 1,
    'Female': 0,
    'male': 1,
    'female': 0
}, inplace=True)

# Convert all columns to numeric
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill missing values
data.fillna(0, inplace=True)

# Features and Target
X = data.drop("classification", axis=1)
y = data["classification"]
print(X.columns)
print(len(X.columns))

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Save Model
pickle.dump(model, open("models/kidney_model.pkl", "wb"))

print("Kidney Model Trained Successfully!")