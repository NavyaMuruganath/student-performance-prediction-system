import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("student_data.csv")

X = data[['G1', 'G2', 'studytime', 'failures', 'absences']]
y = data['G3']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

with open("student_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")

