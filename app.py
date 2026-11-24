
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

model = joblib.load("student_performance_model.pkl")

st.title("🎓 Student Performance Prediction System")
st.write("Predict a student's expected final marks using a Machine Learning model.")

st.divider()

st.subheader("Enter Student Information")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input("Study Hours / Day", min_value=0.0, max_value=16.0, value=5.0, step=0.5)
    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
    previous_marks = st.number_input("Previous Exam Marks", min_value=0.0, max_value=100.0, value=70.0, step=1.0)

with col2:
    assignment_marks = st.number_input("Assignment Marks", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    internal_marks = st.number_input("Internal Assessment Marks", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    completed_assignments = st.number_input("Completed Assignments", min_value=0, max_value=10, value=7, step=1)

if st.button("🔮 Predict Performance", use_container_width=True):
    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "assignment_marks": assignment_marks,
        "internal_marks": internal_marks,
        "completed_assignments": completed_assignments
    }])

    prediction = float(model.predict(input_data)[0])
    prediction = max(0, min(100, prediction))

    if prediction >= 85:
        category = "Excellent 🌟"
    elif prediction >= 70:
        category = "Good 👍"
    elif prediction >= 50:
        category = "Average 📚"
    else:
        category = "Needs Improvement 💡"

    st.divider()
    st.subheader("Prediction Result")
    c1, c2 = st.columns(2)
    c1.metric("Predicted Final Marks", f"{prediction:.1f}/100")
    c2.metric("Performance", category)

    chart_data = pd.DataFrame({
        "Metric": ["Attendance", "Previous Marks", "Assignments", "Internal"],
        "Score": [attendance, previous_marks, assignment_marks, internal_marks]
    })

    fig, ax = plt.subplots()
    ax.bar(chart_data["Metric"], chart_data["Score"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Student Performance Factors")
    plt.xticks(rotation=15)
    st.pyplot(fig)

    st.success("Prediction generated successfully!")

st.divider()
st.caption("BCA AI & DS Mini Project • Student Performance Prediction using Random Forest Regression")
