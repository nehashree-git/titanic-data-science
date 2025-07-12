
import streamlit as st
import pickle
import pandas as pd

# Load model
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚢 Titanic Survival Predictor")

# User Inputs
Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 100, 25)
SibSp = st.number_input("Number of Siblings/Spouses", 0, 10)
Parch = st.number_input("Number of Parents/Children", 0, 10)
Fare = st.number_input("Fare", 0.0, 600.0, 50.0)
Embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# Encoding
sex_map = {"male": 1, "female": 0}
embarked_map = {"S": 2, "C": 0, "Q": 1}

input_df = pd.DataFrame([[
    Pclass,
    sex_map[Sex],
    Age,
    SibSp,
    Parch,
    Fare,
    embarked_map[Embarked]
]], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success("🎉 Survived!" if prediction == 1 else "💀 Did not survive.")
