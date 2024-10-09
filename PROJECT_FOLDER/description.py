import streamlit as st
import pandas as pd
import os 
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, 'datasets')
data = os.path.join(dataset_dir, 'school_dataset.csv')


# Simulate typewriter effect for words with line breaks
def typewriter_effect(text, line_length=90, delay=0.09):
    words = text.split()  # Split text into words
    current_line = ""  # Initialize an empty line

    for word in words:
        if len(current_line) + len(word) + 1 > line_length:  # Check if adding the word exceeds the line length
            st.write(current_line)  # Display the current line
            current_line = word  # Start a new line with the current word
        else:
            current_line += " " + word if current_line else word  # Add the word to the current line
        
        time.sleep(delay)  # Add delay between words

    if current_line:  # Ensure the last line is also printed
        st.write(current_line)

def describing():
    # Load your data
    df = pd.read_csv(data)

    # Descriptive statistics section
    st.header("Descriptive Statistics")

    # General overview of the dataset
    intro_text = (f"We have {df.shape[0]} rows of student's information in the database and "
                  f"there are {df['student_id'].nunique()} students in the school. "
                  f"Some of the columns include {df.columns.to_list()[:10]}.\n")

    typewriter_effect(intro_text, line_length=120)  # Customize line length and delay

    # Distribution of numerical features
    st.subheader("Distribution of Numerical Features")
    final_grade_text = (f"The average final grade is {df['final_grade'].mean():.2f} with a standard deviation of {df['final_grade'].std():.2f}. "
                        f"The final grades range from {df['final_grade'].min()} to {df['final_grade'].max()}.\n")
    
    typewriter_effect(final_grade_text, line_length=120)

    exam_score_text = (f"The average exam score is {df['exam_score'].mean():.2f} with a standard deviation of {df['exam_score'].std():.2f}. "
                       f"The exam scores range from {df['exam_score'].min()} to {df['exam_score'].max()}.\n")
    
    typewriter_effect(exam_score_text, line_length=120)

    # You can continue this for all parts of your output where you want the text to appear word by word.

    st.subheader("Correlation with Target Variable")
    df['target'] = df['target'].map({'Pass': 1, 'Fail': 0})
    correlation_matrix = df.select_dtypes('number').corr()
    target_correlation = correlation_matrix['target'].sort_values(ascending=False)[1:4]

    findings_text = ("The top 3 columns that are most correlated with the target variable are as follows:\n")
    typewriter_effect(findings_text, line_length=120)

    for index, value in target_correlation.items():
        correlation_text = f"- {index}: {value:.2f}\n"
        typewriter_effect(correlation_text, line_length=120)

# Call your function to render the page
# describing()


# def describing():
#     # Load your data
#     df = pd.read_csv(data)

#     # Descriptive statistics section
#     st.header("Descriptive Statistics")

#     # General overview of the dataset
#     st.write(f"We have {df.shape[0]} rows of student's information in the database and\
#             there are {df['student_id'].nunique()} students in the school")
#     st.write(f"Some of the columns include {df.columns.to_list()[:10]}.")

#     # Distribution of numerical features
#     st.subheader("Distribution of Numerical Features")
#     st.write("### Final Grade")
#     st.write(f"The average final grade is {df['final_grade'].mean():.2f} with a standard deviation of {df['final_grade'].std():.2f}.")
#     st.write(f"The final grades range from {df['final_grade'].min()} to {df['final_grade'].max()}.")

#     st.write("### Exam Score")
#     st.write(f"The average exam score is {df['exam_score'].mean():.2f} with a standard deviation of {df['exam_score'].std():.2f}.")
#     st.write(f"The exam scores range from {df['exam_score'].min()} to {df['exam_score'].max()}.")

#     st.write("### Age")
#     st.write(f"The average age of students is {df['age'].mean():.2f} years with a standard deviation of {df['age'].std():.2f}.")
#     st.write(f"The ages range from {df['age'].min()} to {df['age'].max()} years.")

#     # Analysis of categorical variables
#     st.subheader("Analysis of Categorical Variables")
#     st.write("### Class Distribution")
#     class_counts = df['class'].value_counts()
#     st.write("The class distribution is as follows:")
#     for index, value in class_counts.items():
#         st.write(f"- {index}: {value} students")

#     st.write("### Gender Distribution")
#     gender_counts = df['gender'].value_counts()
#     st.write("The gender distribution is as follows:")
#     for index, value in gender_counts.items():
#         st.write(f"- {index}: {value} students")

#     st.write("### Pass/Fail Distribution")
#     target_counts = df['target'].value_counts()
#     st.write("The distribution of pass/fail status is as follows:")
#     for index, value in target_counts.items():
#         st.write(f"- {index}: {value} students")

#     # Correlation with target variable
#     st.subheader("Correlation with Target Variable")
#     df['target'] = df['target'].map({'Pass':1,'Fail':0})
#     correlation_matrix = df.select_dtypes('number').corr()
#     target_correlation = correlation_matrix['target'].sort_values(ascending=False).head(5)
#     st.write("The top 5 columns that are most correlated with the target variable are as follows:")
#     for index, value in target_correlation.items():
#         st.write(f"- {index}: {value:.2f}")

#     # Key findings
#     st.subheader("Key Findings")
#     st.write(f"1. There is a strong correlation between {target_correlation.index[2]} and the likelihood of passing.")
#     st.write("2. Gender distribution is relatively balanced, but slight variations may exist in performance.")
#     st.write("3. Certain classes have higher average scores than others, indicating potential disparities in teaching effectiveness.")
#     st.write("4. Age shows some influence on exam performance, with older students generally performing better.")
