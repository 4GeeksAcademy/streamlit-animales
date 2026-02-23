import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from pickle import dump
import json

url = "https://data.austintexas.gov/api/views/9t4d-g238/rows.csv?accessType=DOWNLOAD"
df = pd.read_csv(url)

df = df[["Animal Type", "Sex upon Outcome", "Age upon Outcome", "Outcome Type"]].dropna()

# solo me interesa perro, gato o cualquier otro
df["Animal Type"] = df["Animal Type"].apply(
    lambda x: x if x in ["Dog", "Cat"] else "Other"
)

le_sex = LabelEncoder()
le_age = LabelEncoder()
le_outcome = LabelEncoder()
le_target = LabelEncoder()

df["Sex"] = le_sex.fit_transform(df["Sex upon Outcome"])
df["Age"] = le_age.fit_transform(df["Age upon Outcome"])
df["Outcome"] = le_outcome.fit_transform(df["Outcome Type"])
df["Target"] = le_target.fit_transform(df["Animal Type"])

X = df[["Sex", "Age", "Outcome"]]
y = df["Target"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

dump(model, open("../models/model.pkl", "wb"))
dump(le_sex, open("../models/le_sex.pkl", "wb"))
dump(le_age, open("../models/le_age.pkl", "wb"))
dump(le_outcome, open("../models/le_outcome.pkl", "wb"))
dump(le_target, open("../models/le_target.pkl", "wb"))

categories = {
    "sex": list(le_sex.classes_),
    "age": list(le_age.classes_),
    "outcome": list(le_outcome.classes_)
}

with open("../models/categories.json", "w") as f:
    json.dump(categories, f)

print("Listo, modelo guardado.")
print("Clases:", list(le_target.classes_))
