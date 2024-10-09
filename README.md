## Student Academic Performance Prediction Project

You can click on this link to  run our streamlit app

[a streamlitapp](https://arrow-team-student-performance-st.streamlit.app/)

## Project Overview
This project focuses on predicting student academic performance using machine learning models and conducting data analysis on student records. The project aims to provide actionable insights to educators, administrators, and stakeholders to improve student outcomes. We utilize a variety of machine learning techniques and present results through an interactive web application developed with Streamlit.

Additionally, the project includes visual data analysis and a barcode attendance tracking system, creating a comprehensive solution for academic performance monitoring.

## Key Features

- Data Analysis: Comprehensive exploratory data analysis (EDA) on student-related features such as exam scores, final grades, class, gender, and ethnicity.
- Model Training: A machine learning model, primarily based on transformer architecture, was trained to predict students' pass/fail outcomes based on various input features.
- Streamlit Application: A user-friendly interface for interacting with data, visualizing student performance, and displaying model predictions.
- Barcode Attendance Tracker: A system to track student attendance via barcode scanning, linked to academic performance prediction.

Prerequisites
Make sure you have the following installed:

Python 3.7+
Required packages listed in requirements.txt:
bash
Copy code
pandas
numpy
scikit-learn
torch
streamlit
plotly
matplotlib

Install the dependencies in the requirements.txt

The dataset used in this project is a randomly generated synthetic dataset (school_dataset.csv) that mimics real-world student records. The key features include:

- Class: The grade/class of the student.
- Gender: The gender of the student.
- Exam Score: The score a student received in an exam.
-Final Grade: The final grade for the academic year.
- Ethnicity: Ethnic background of the student.
- Disability Status: Whether the student has a disability or not.
- Target (Pass/Fail): The outcome variable indicating whether the student passed or failed.
  
## Data Analysis
The exploratory data analysis (EDA) provides insights into the distribution of key features, such as exam scores and final grades, and how they relate to the target variable (pass/fail). The analysis was performed in the Jupyter notebook located in the notebooks/ directory.

## Key Insights:
Gender Distribution: Visualized with bar plots, showing gender-based performance trends.
Class and Exam Scores: Scatter plots illustrating relationships between exam scores and final grades across different classes.
Pass/Fail Rates: Class-level success and failure rates visualized with funnel and sunburst charts.
Model Training
The machine learning model used for predicting academic performance is a transformer-based deep learning model. It uses attention mechanisms to weigh the importance of various features, such as exam scores and class, in predicting whether a student will pass or fail.



## Streamlit Application
The Streamlit app provides an intuitive interface for users to interact with the data and model predictions. The app is located in the app/ directory and includes multiple data visualizations (bar charts, sunburst charts, scatter plots, etc.) that help stakeholders understand student performance trends.


# Run the Streamlit app
streamlit run app.py
The app will open in your web browser, where you can explore various charts and visualizations. It also features a section where users can upload new student data for prediction, with results displayed in real time.
