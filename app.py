import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt


model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title(" Student Performance Prediction App")


student_id = st.text_input("Student ID")

age = st.number_input("Age", 12, 30)
study_hours = st.slider("Study Hours", 0, 12)
attendance = st.slider("Attendance %", 0, 100)
assignments = st.slider("Assignments Completed", 0, 10)
midterm = st.slider("Midterm Marks", 0, 100)
final_exam = st.slider("Final Exam Marks", 0, 100)

gender = st.selectbox("Gender", ["Male", "Female"])
internet = st.selectbox("Internet Access", ["Yes", "No"])


if st.button("Predict Performance"):

    
    input_data = pd.DataFrame({
        "age": [age],
        "study_hours": [study_hours],
        "attendance": [attendance],
        "assignments": [assignments],
        "midterm": [midterm],
        "final_exam": [final_exam],
        "gender": [gender],
        "internet": [internet]
    })

    
    input_data = pd.get_dummies(input_data)

    
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[columns]

    
    prediction = model.predict(input_data)[0]

    
    st.subheader(" Prediction Result")

    if prediction == 1:
        st.success(" Student is Likely to PASS")
        risk = "LOW RISK"
    else:
        st.error(" Student is Likely to FAIL")
        risk = "HIGH RISK"

    st.write(" Risk Level:", risk)

    st.subheader(" Performance Dashboard")

    labels = ["Study", "Attendance", "Assignments", "Midterm", "Final"]
    values = [study_hours*8, attendance, assignments*10, midterm, final_exam]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Score")
    ax.set_title("Student Performance Indicators")

    st.pyplot(fig)
