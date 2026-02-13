import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


data = {
    "age": [18,19,20,21,22,23,20,19,18,22],
    "study_hours": [2,5,6,8,1,3,7,4,9,2],
    "attendance": [60,80,90,95,50,70,85,75,98,55],
    "assignments": [4,7,8,9,2,5,8,6,10,3],
    "midterm": [50,70,80,90,40,60,85,65,95,45],
    "final_exam": [55,75,85,92,45,65,88,70,97,50],
    "gender": ["Male","Female","Male","Female","Male","Female","Male","Female","Male","Female"],
    "internet": ["Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No"],
    "result": [0,1,1,1,0,1,1,1,1,0]
}

df = pd.DataFrame(data)


X = df.drop("result", axis=1)
y = df["result"]


X = pd.get_dummies(X)

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))


pickle.dump(X.columns, open("columns.pkl", "wb"))

print("✅ Model and Columns saved successfully!")
