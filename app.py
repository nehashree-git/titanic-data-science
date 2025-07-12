import streamlit as st
import pickle
import pandas as pd

# Load the model
try:
    with open("logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# App Title
st.title("🚢 Titanic Survival Predictor")

st.markdown(
    """
    Enter passenger details below and click **Predict** to see survival prediction.
    """
)

# User Inputs
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 1, 100, 25)
SibSp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, step=1)
Parch = st.number_input("Number of Parents/Children Aboard", 0, 10, step=1)
Fare = st.number_input("Fare", 0.0, 600.0, step=1.0)
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode categorical inputs
sex_map = {"male": 1, "female": 0}
embarked_map = {"C": 0, "Q": 1, "S": 2}

# Create input DataFrame
input_data = pd.DataFrame([[
    Pclass,
    sex_map[Sex],
    Age,
    SibSp,
    Parch,
    Fare,
    embarked_map[Embarked]
]], columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])

# Prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.success("🎉 The passenger would have **Survived**.")
        else:
            st.error("💀 The passenger would **Not Survive**.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
