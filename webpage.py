import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# creating function for prediction
def heart_disease_pred(input_data):
    input_data_numpy_array = np.asarray(input_data)
    # reshaping numpy array
    input_data_rshape = input_data_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_rshape)
    print(prediction)

    if prediction[0] == 0:
        return "Person does not have heart disease"
    else:
        return "Person is suffering from heart disease"


def main():
    # Giving title for webpage
    st.title("Heart Disease Prediction web app")

    # Taking input from user
    age = st.text_input("Age of patient")
    sex = st.text_input("Gender of patient(0--->Female,1--->Male)")
    cp = st.text_input("Chest pain type(Value 1: typical angina ,Value 2: atypical angina, Value 3: non-anginal pain , Value 4: asymptomatic)")
    trestbps = st.text_input("Resting blood pressure")
    chol = st.text_input("Serum cholestrol in mg/dl")
    fbs = st.text_input("Fasting blood sugar > 120 mg/dl(1--->True,0---->False) ")
    restecg = st.text_input("resting electrocardiographic results (values 0,1,2)")
    thalach = st.text_input("maximum heart rate achieved")
    exang = st.text_input("exercise induced angina(1--->Yes,0--->No)")
    oldpeak = st.text_input("ST depression induced by exercise relative to rest")
    slope = st.text_input("slope of the peak exercise ST segment(Value 1: up sloping , Value 2: flat , Value 3: down sloping )")
    ca = st.text_input("number of major vessels (0-3) colored by flourosopy")
    thal = st.text_input("thal: 3 = normal; 6 = fixed defect; 7 = reversable defect")

    # code for prediction
    diagnosis = ""

    # creating button for prediction
    if st.button("Predict result"):
        diagnosis = heart_disease_pred(
            [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
    st.success(diagnosis)


if __name__ == '__main__':
    main()
