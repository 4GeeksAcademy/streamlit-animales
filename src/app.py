import streamlit as st
from pickle import load
import json

model = load(open("../models/model.pkl", "rb"))
le_sex = load(open("../models/le_sex.pkl", "rb"))
le_age = load(open("../models/le_age.pkl", "rb"))
le_condition = load(open("../models/le_condition.pkl", "rb"))
le_target = load(open("../models/le_target.pkl", "rb"))

with open("../models/categories.json", "r") as f:
    categories = json.load(f)

st.title("Prediccion de tipo de animal")
st.write("Completa los datos del animal para saber si es un perro, gato u otro.")

sex = st.selectbox("Sexo al momento de ingreso", categories["sex"])
age = st.selectbox("Edad al momento de ingreso", sorted(categories["age"]))
condition = st.selectbox("Condicion fisica al ingreso", categories["condition"])

if st.button("Predecir"):
    sex_enc = le_sex.transform([sex])[0]
    age_enc = le_age.transform([age])[0]
    condition_enc = le_condition.transform([condition])[0]

    pred = model.predict([[sex_enc, age_enc, condition_enc]])[0]
    animal = le_target.inverse_transform([pred])[0]

    if animal == "Dog":
        st.success("El animal es probablemente un Perro.")
    elif animal == "Cat":
        st.success("El animal es probablemente un Gato.")
    else:
        st.success("El animal es de otra especie.")
