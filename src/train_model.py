import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from pickle import dump
import json

url = "https://data.austintexas.gov/api/views/9t4d-g238/rows.csv?accessType=DOWNLOAD"
df = pd.read_csv(url)

df = df[["Animal Type", "Sex upon Intake", "Age upon Intake", "Intake Condition"]].dropna()

# solo me interesa perro, gato o cualquier otro
df["Animal Type"] = df["Animal Type"].apply(
    lambda x: x if x in ["Dog", "Cat"] else "Other"
)

le_sex = LabelEncoder()
le_age = LabelEncoder()
le_condition = LabelEncoder()
le_target = LabelEncoder()

df["Sex"] = le_sex.fit_transform(df["Sex upon Intake"])
df["Age"] = le_age.fit_transform(df["Age upon Intake"])
df["Condition"] = le_condition.fit_transform(df["Intake Condition"])
df["Target"] = le_target.fit_transform(df["Animal Type"])

X = df[["Sex", "Age", "Condition"]]
y = df["Target"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

dump(model, open("../models/model.pkl", "wb"))
dump(le_sex, open("../models/le_sex.pkl", "wb"))
dump(le_age, open("../models/le_age.pkl", "wb"))
dump(le_condition, open("../models/le_condition.pkl", "wb"))
dump(le_target, open("../models/le_target.pkl", "wb"))

categories = {
    "sex": list(le_sex.classes_),
    "age": list(le_age.classes_),
    "condition": list(le_condition.classes_)
}

with open("../models/categories.json", "w") as f:
    json.dump(categories, f)

print("Listo, modelo guardado.")
print("Clases:", list(le_target.classes_))
