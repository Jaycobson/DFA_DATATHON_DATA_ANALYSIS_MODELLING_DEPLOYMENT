import os
import pandas as pd
import psycopg2
import numpy as np

def getting():
    # Establish the database connection (replace with your credentials)
    connection = psycopg2.connect(
        user="avnadmin",              
        password="AVNS_blUS8t5v_YlvF0J_omz",         
        host="pg-353bb115-adeitanemmanuel086-380.h.aivencloud.com",            
        port="26014",                  
        database="defaultdb",    
        sslmode="require"
    )
    # Load your dataset (adjust path accordingly)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, 'datasets')
    os.makedirs(dataset_dir, exist_ok=True)

    # SQL queries and writing to CSV
    queries = {
        'students': "SELECT * FROM students",
        'parents': "SELECT * FROM parents",
        'teachers': "SELECT * FROM teachers",
        'attendance': "SELECT * FROM attendance",
        'attendance_summary': "SELECT * FROM attendance_summary",
        'student_academic_data': "SELECT * FROM student_academic_data",
        'aggregated_academic': "SELECT * FROM aggregated_academic",
        'teacher_evaluation_data': "SELECT * FROM teacher_evaluation_data",
        'aggregated_teacher_evaluation': "SELECT * FROM aggregated_teacher_evaluation",
        'class_to_teacher': "SELECT * FROM class_to_teacher",
        'subjects_offered': "SELECT * FROM subjects_offered",
        # 'final': """
        #     SELECT s.student_id, s.student_name, s.class, p.parent_name, aa.average_final_grade, te.average_teacher_evaluation
        #     FROM students s
        #     LEFT JOIN parents p ON s.student_id = p.student_id
        #     LEFT JOIN aggregated_academic aa ON s.student_id = aa.student_id
        #     LEFT JOIN aggregated_teacher_evaluation te ON s.student_id = te.student_id;
        # """
    }

    # Dictionary to store DataFrames dynamically
    dataframes = {}

    # Iterate over each query, execute it, and save the result as CSV
    for name, query in queries.items():
        # Read SQL query into a pandas DataFrame and store it in the dictionary
        dataframes[name] = pd.read_sql(query, connection)
        
        # Define the path for saving the CSV file
        file_path = os.path.join(dataset_dir,f'{name}_data.csv')
        
        # Save the DataFrame to a CSV file
        dataframes[name].to_csv(file_path, index=False)

    merged_df = pd.merge(dataframes['students'], dataframes['parents'], on='student_id', how='left',  suffixes=('', '_parent'))
    merged_df = pd.merge(merged_df, dataframes['student_academic_data'], on='student_id', how='left',  suffixes=('', '_academic'))
    merged_df = pd.merge(merged_df, dataframes['teacher_evaluation_data'], on=['student_id','subject_id'], how='inner',  suffixes=('', '_teacher_evaluation'))
    merged_df = pd.merge(merged_df, dataframes['subjects_offered'], on=['subject_id'], how='left',  suffixes=('', '_subject'))
    # merged_df = pd.merge(merged_df, class_assignment_df, on=['class','subject_id'], how='left',  suffixes=('', '_class'))
    # merged_df = pd.merge(merged_df, teachers_df, on='teacher_id', how='inner',  suffixes=('', '_teacher'))

    final_df = pd.merge(merged_df, dataframes['attendance_summary'], on='student_id', how='left',  suffixes=('', '_attendance_summary'))


    # Define thresholds
    attendance_threshold = 50
    exam_score_threshold = 50
    assignment_score_threshold = 50
    final_grade_threshold = 50

    # You can add more complex conditions based on study time, anxiety, etc.final
    final_df['target'] = np.where(
        (final_df['attendance_percentage'] >= attendance_threshold) & (final_df['point'] > 5) & (final_df['internetaccess'] == True) & (final_df['access_to_constant_electricity'] == True) & (final_df['anxiety_before_during_exams'] == False) &
        (final_df['exam_score'] >= exam_score_threshold) & 
        (final_df['assignment_score'] >= assignment_score_threshold) &
        (final_df['final_grade'] >= final_grade_threshold), 
        'Pass', 
        'Fail'
    )

    # # Optional: Add more conditions if needed
    final_df['target'] = np.where(
        (final_df['attendance_percentage'] < attendance_threshold) | 
        (final_df['exam_score'] < exam_score_threshold) |
        (final_df['final_grade'] < final_grade_threshold) |
        (final_df['assignment_score'] < assignment_score_threshold), 
        'Fail', 
        'Pass')


    file_path = os.path.join(dataset_dir, f'school_dataset.csv')
    final_df.to_csv(file_path, index=False)


    # Close the database connection
    connection.close()



