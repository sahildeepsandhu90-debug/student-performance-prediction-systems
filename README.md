# Student Performance Prediction System

## How to run

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   pip install -r requirements.txt

4. Start the app:

   streamlit run app.py

5. Open the local address shown by Streamlit in your browser.

## Files

- app.py — Streamlit web application
- student_data.csv — sample training dataset
- student_performance_model.pkl — trained Random Forest model
- requirements.txt — required Python packages

## ML model

Random Forest Regression is used to predict final marks from study hours, attendance, previous marks, assignment marks, internal assessment marks, and completed assignments.
