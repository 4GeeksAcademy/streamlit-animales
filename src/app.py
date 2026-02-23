import os
import json
import streamlit as st
import pandas as pd
from pickle import load, dump
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_save():
    url = "https://data.austintexas.gov/api/views/9t4d-g238/rows.csv?accessType=DOWNLOAD"
    df = pd.read_csv(url)
    df = df[["Animal Type", "Sex upon Outcome", "Age upon Outcome", "Outcome Type"]].dropna()
    df["Animal Type"] = df["Animal Type"].apply(lambda x: x if x in ["Dog", "Cat"] else "Other")

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

    dump(model, open(os.path.join(MODELS_DIR, "model.pkl"), "wb"))
    dump(le_sex, open(os.path.join(MODELS_DIR, "le_sex.pkl"), "wb"))
    dump(le_age, open(os.path.join(MODELS_DIR, "le_age.pkl"), "wb"))
    dump(le_outcome, open(os.path.join(MODELS_DIR, "le_outcome.pkl"), "wb"))
    dump(le_target, open(os.path.join(MODELS_DIR, "le_target.pkl"), "wb"))

    categories = {
        "sex": list(le_sex.classes_),
        "age": list(le_age.classes_),
        "outcome": list(le_outcome.classes_)
    }
    with open(os.path.join(MODELS_DIR, "categories.json"), "w") as f:
        json.dump(categories, f)

if not os.path.exists(os.path.join(MODELS_DIR, "model.pkl")):
    with st.spinner("Preparando el modelo, espera un momento..."):
        train_and_save()

model = load(open(os.path.join(MODELS_DIR, "model.pkl"), "rb"))
le_sex = load(open(os.path.join(MODELS_DIR, "le_sex.pkl"), "rb"))
le_age = load(open(os.path.join(MODELS_DIR, "le_age.pkl"), "rb"))
le_outcome = load(open(os.path.join(MODELS_DIR, "le_outcome.pkl"), "rb"))
le_target = load(open(os.path.join(MODELS_DIR, "le_target.pkl"), "rb"))

with open(os.path.join(MODELS_DIR, "categories.json"), "r") as f:
    categories = json.load(f)

st.title("Prediccion de tipo de animal")
st.write("Completa los datos del animal para saber si es un perro, gato u otro.")

sex = st.selectbox("Sexo del animal", categories["sex"])
age = st.selectbox("Edad del animal", sorted(categories["age"]))
outcome = st.selectbox("Tipo de resultado", categories["outcome"])

if st.button("Predecir"):
    sex_enc = le_sex.transform([sex])[0]
    age_enc = le_age.transform([age])[0]
    outcome_enc = le_outcome.transform([outcome])[0]

    pred = model.predict([[sex_enc, age_enc, outcome_enc]])[0]
    animal = le_target.inverse_transform([pred])[0]

    if animal == "Dog":
        st.success("El animal es probablemente un Perro.")
    elif animal == "Cat":
        st.success("El animal es probablemente un Gato.")
    else:
        st.success("El animal es de otra especie.")
