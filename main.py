import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("titanic.csv")

print("📂 First 5 rows of data:")
print(df.head())
# Drop rows with missing values (simplest way)
df = df.dropna(subset=["Age", "Embarked"])

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

sample_passenger = np.array([[3, 1, 22, 0, 0, 7.25, 2]])
prediction = model.predict(sample_passenger)

print("\n🚢 Prediction: Survived" if prediction[0] == 1 else "\n⚓ Prediction: Did not survive")
