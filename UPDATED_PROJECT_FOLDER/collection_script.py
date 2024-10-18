import streamlit as st
import pandas as pd
import psycopg2
import numpy as np

# PostgreSQL connection setup
connection = psycopg2.connect(
    user="avnadmin",              
    password="AVNS_blUS8t5v_YlvF0J_omz",         
    host="pg-353bb115-adeitanemmanuel086-380.h.aivencloud.com",            
    port="26014",                  
    database="defaultdb",    
    sslmode="require"
)

# Helper function to generate unique student IDs
def collecting_student_info():
    subjects = [
    'Mathematics', 'English', 'Science', 'History', 
    'Geography', 'Physics', 'Chemistry', 'Biology', 'Economics'

]
    st.title("Student Information Collection Form")

    with st.form("student_form", clear_on_submit=True):
        # Personal Details Section
        st.header("Personal Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            student_id = st.text_input('Enter Student ID')
        with col2:
            student_name = st.text_input("Student Name", max_chars=50)
        with col3:
            class_type = st.selectbox("Class", ['Jss 1', 'Jss 2', 'Jss 3', 'SS1', 'SS2', 'SS3'])

        col4, col5 = st.columns(2)
        with col4:
            age = st.slider("Age", min_value=10, max_value=20)
            gender = st.radio("Gender", ['Male', 'Female'], index=1)
        with col5:
            ethnicity = st.selectbox("Ethnicity", ["Hausa", "Igbo", "Yoruba", "Other"])

        # Family & Academic Information
        st.header("Family & Academic Information")
        col6, col7 = st.columns(2)
        with col6:
            disability_status = st.checkbox("Do you have a disability?")
            family_size = st.selectbox("Family Size", ['0', '1', '2', '3', '4', 'more than 5'])
        with col7:
            favorite_subject = st.selectbox("Favorite Subject", subjects)
            access_to_constant_electricity = st.radio(
                "Do you have access to constant electricity?", [True, False]
            )
            on_scholarship = st.radio("Are you on a scholarship?", [True, False])

        # Daily Routines
        st.header("Daily Routines")
        col8, col9 = st.columns(2)
        with col8:
            average_study_time = st.selectbox(
                "How many hours do you study daily?", 
                ['<1 hour', '1 - 2 hours', '2 - 3 hours', '3 - 4 hours', '4 - 6 hours', '> 6 hours']
            )
        with col9:
            sleep = st.selectbox("How many hours do you sleep on average?", ['<6 hours', '6 - 8 hours', '>8 hours'])

        # Activities & Social Habits
        st.header("Activities & Social Habits")
        col10, col11 = st.columns(2)
        with col10:
            anxiety_before_during_exams = st.checkbox("Do you experience anxiety during exams?")
            enjoy_reading = st.checkbox("Do you enjoy reading?")
        with col11:
            enjoy_dancing = st.checkbox("Do you enjoy dancing?")
            enjoy_socialising = st.checkbox("Do you enjoy socialising with others?")

        # School Logistics
        st.header("School Logistics")
        col12, col13 = st.columns(2)
        with col12:
            distance_to_school = st.selectbox("How far is your school from home?", 
                                              ['very close', 'close', 'far', 'very far'])
        with col13:
            mode_of_transportation = st.selectbox(
                "Mode of Transportation to School", 
                ['trekking', 'bicycle', 'tricycle', 'school bus', 'private/family car']
            )

        # Submit Button
        submitted = st.form_submit_button("Submit")
       
    # Process and insert the submitted data
    if submitted:
        st.success('Successfully added!')
        # Create a dictionary for the student data
        student_data = {
            'student_id': student_id,
            'student_name': student_name,
            'class': class_type,
            'age': age,
            'gender': gender,
            'ethnicity': ethnicity,
            'disability_status': disability_status,
            'family_size': family_size,
            'favorite_subjects': favorite_subject,
            'access_to_constant_electricity': access_to_constant_electricity,
            'on_scholarship': on_scholarship,
            'average_study_time': average_study_time,
            'sleep': sleep,
            'anxiety_before_during_exams': anxiety_before_during_exams,
            'enjoy_reading': enjoy_reading,
            'enjoy_dancing': enjoy_dancing,
            'enjoy_socialising': enjoy_socialising,
            'distance_to_school': distance_to_school,
            'mode_of_transportation': mode_of_transportation
        }

        # Insert data into the database (mocked here as a print statement)
        insert_query = """
            INSERT INTO students (
                student_id, student_name, class, age, gender, ethnicity, disability_status,
                family_size, favorite_subjects, access_to_constant_electricity, on_scholarship,
                average_study_time, sleep, anxiety_before_during_exams, enjoy_reading, enjoy_dancing,
                enjoy_socialising, distance_to_school, mode_of_transportation
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        values = tuple(student_data.values())

        try:
            with connection.cursor() as cursor:
                cursor.execute(insert_query, values)
                connection.commit()
                st.success("Student Information Submitted and Saved to Database Successfully!")
        except Exception as e:
            st.error(f"An error occurred: {e}")

        # Display the submitted data
        student_df = pd.DataFrame([student_data])
        st.dataframe(student_df)

def collecting_parent_info():
    st.title("Parent/Guardian Information Collection")

    with st.form("parent_form", clear_on_submit=True):
        st.subheader("Personal & Contact Information")

        # Use columns to speed up input
        col1, col2 = st.columns(2)
        with col1:
            parent_id = st.text_input("Parent ID")
            student_id = st.text_input("Associated Student ID")
            parent_name = st.text_input("Parent/Guardian Name", max_chars=50)
            gender = st.radio("Gender", ['Male', 'Female'], index=0)
        with col2:
            email = st.text_input("Email Address")
            phone_number = st.text_input("Phone Number")

        st.subheader("Family & Financial Information")
        col3, col4 = st.columns(2)
        with col3:
            family_income_range = st.selectbox(
                "Family Income Range", 
                ['<#30,000', '#30,001 - #70,000', '#70,001 - #120,000', '#120,001 - #200,000', '>#200,001']
            )
            number_of_children = st.selectbox("Number of Children", ['0', '1', '2', '3', '4', '5', 'more than 5'])
            house_ownership = st.radio("House Ownership", ['Rent', 'Owned'])
        with col4:
            occupation = st.selectbox("Occupation", ['Unemployed', 'Self Employed', 'Employed'])

        st.subheader("Education & Work Information")
        col5, col6 = st.columns(2)
        with col5:
            education_level = st.selectbox(
                "Education Level", 
                ['High School', 'Associate Degree', 'Bachelor\'s Degree', 'Master\'s Degree', 'PhD']
            )
            marital_status = st.radio("Marital Status", ['Married', 'Single', 'Divorced', 'Widowed'])
        with col6:
            location = st.selectbox("Location", ['Urban', 'Suburban', 'Rural'])
            age_range = st.selectbox("Age Range", ['25-34', '35-44', '45-54', '55-64', 'Older'])
            work_hours_per_week = st.slider("Work Hours per Week", min_value=0, max_value=60, step=1)

        st.subheader("Employment & Transportation")
        col7, col8 = st.columns(2)
        with col7:
            employment_status = st.radio("Employment Status", ['Full-time', 'Part-time', 'Freelance'])
        with col8:
            transportation_mode = st.selectbox("Mode of Transportation", ['Car', 'Bus', 'Bike', 'Trekking'])

        st.subheader("Access & Qualifications")
        col9, col10 = st.columns(2)
        with col9:
            internet_access = st.radio("Internet Access at Home?", [True, False])
        with col10:
            qualification = st.selectbox(
                "Qualification", 
                ['Bachelors', 'Masters', 'PhD', 'Diploma', 'None']
            )

        submitted = st.form_submit_button("Submit")

        if submitted:
            parent_data = {
                'parent_id': parent_id,
                'student_id': student_id,
                'parent_name': parent_name,
                'gender': gender,
                'email': email,
                'familyIncomeRange': family_income_range,
                'numberOfChildren': number_of_children,
                'occupation': occupation,
                'houseOwnership': house_ownership,
                'educationLevel': education_level,
                'maritalStatus': marital_status,
                'location': location,
                'age': age_range,
                'workHoursPerWeek': work_hours_per_week,
                'employmentStatus': employment_status,
                'transportationMode': transportation_mode,
                'internetAccess': internet_access,
                'qualification': qualification,
                'phone_number': phone_number
            }


            # SQL Insert Query for Parent Data
            insert_parent_query = """
                INSERT INTO parents (
                    parent_id, student_id, parent_name, gender, email, familyIncomeRange, numberOfChildren,
                    occupation, houseOwnership, educationLevel, maritalStatus, location, age,
                    workHoursPerWeek, employmentStatus, transportationMode, internetAccess, qualification, phone_number
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            # Extract the values from the parent_data dictionary
            parent_values = tuple(parent_data.values())

            # Insert parent data into the database
            try:
                with connection.cursor() as cursor:
                    cursor.execute(insert_parent_query, parent_values)
                    connection.commit()
                    st.success("Parent Information Submitted and Saved to Database Successfully!")
            except Exception as e:
                st.error(f"An error occurred: {e}")

            # Save to DataFrame
            parent_df = pd.DataFrame([parent_data])

            # Display Success Message and Data
            st.success("Parent/Guardian Information Submitted Successfully!")
            st.dataframe(parent_df)


def collecting_teacher_info():
    st.title("Teacher Information Collection")

    with st.form("teacher_form", clear_on_submit=True):
        st.subheader("Teacher Personal & Contact Information")

        col1, col2 = st.columns(2)
        with col1:
            teacher_id = st.text_input("Teacher ID")
            teacher_name = st.text_input("Teacher Name", max_chars=50)
            gender = st.radio("Gender", ['Male', 'Female'], index=0)
            phone_number = st.text_input("Phone Number")
        with col2:
            email = st.text_input("Email Address")
            years_of_experience = st.slider("Years of Experience", min_value=1, max_value=30)

        st.subheader("Professional Information")

        col3, col4 = st.columns(2)
        with col3:
            department = st.selectbox("Department", ['Mathematics', 'Science', 'English', 'History', 'Art'])
            salary_range = st.selectbox(
                "Salary Range", 
                ['<#50,000', '#50,001 - #100,000', '#100,000 - #150,000', '#150,000+']
            )
        with col4:
            employment_status = st.radio("Employment Status", ['Full-time', 'Part-time'])
            qualification = st.selectbox("Qualification", ['Bachelors', 'Masters', 'PhD', 'Diploma'])

        submitted = st.form_submit_button("Submit")

        if submitted:
            teacher_data = {
                'teacher_id': teacher_id,
                'teacher_name': teacher_name,
                
    
                'teacher_email': email,
                'salary_range': salary_range,
              
                'department': department,
                'qualification': qualification,
                'employmentStatus': employment_status,
                
                'gender': gender,
                'yearsOfExperience': years_of_experience,
                'phone_number': phone_number,
            }
    # Prepare SQL Insert Query for Teacher Data
            insert_teacher_query = """
                INSERT INTO teachers (
                    teacher_id, teacher_name, teacher_email, salary_range, department, qualification,
                    employmentStatus, gender, yearsOfExperience, phone_number
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            # Extract the values from the teacher_data dictionary
            teacher_values = tuple(teacher_data.values())

            # Insert teacher data into the database
            try:
                with connection.cursor() as cursor:
                    cursor.execute(insert_teacher_query, teacher_values)
                    connection.commit()
                    st.success("Teacher Information Submitted and Saved to Database Successfully!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
            # Save to DataFrame
            teacher_df = pd.DataFrame([teacher_data])

            # Display Success Message and Data
            st.success("Teacher Information Submitted Successfully!")
            st.dataframe(teacher_df)


def collect_subject_info():
    st.title("Subject Information Collection")
    subject_data = []

    with st.form("subject_form", clear_on_submit=True):
        subject_id = st.text_input("Subject ID")
        subject_name = st.text_input("Subject Name")

        submitted = st.form_submit_button("Submit")
        if submitted:
            subject_data.append({"subject_id": subject_id, "subject_name": subject_name})
            st.success("Subject Information Submitted Successfully!")
            

            insert_subject_query = """
            INSERT INTO subjects (
                subject_id, subject_name
            ) VALUES (%s, %s)
        """

            # Create a dictionary for the subject data
            subject_data = {
                "subject_id": subject_id,
                "subject_name": subject_name
            }

            # Extract values from the dictionary
            subject_values = tuple(subject_data.values())

            # Insert subject data into the database
            try:
                with connection.cursor() as cursor:
                    cursor.execute(insert_subject_query, subject_values)
                    connection.commit()
                    st.success("Subject Information Submitted and Saved to Database Successfully!")
            except Exception as e:
                st.error(f"An error occurred: {e}")

            subject_df = pd.DataFrame(subject_data)
            st.dataframe(subject_df)

def collect_attendance_info():
    st.title("Attendance Record Collection")
    attendance_data = []

    with st.form("attendance_form", clear_on_submit=True):
        attendance_id = st.text_input("Attendance ID")
        student_id = st.text_input("Student ID")
        date = st.date_input("Date")
        status = st.radio("Attendance Status", ['Present', 'Absent'])

        submitted = st.form_submit_button("Submit")
        if submitted:
            status_value = 1 if status == 'Present' else 0
            attendance_data.append({
                "attendance_id": attendance_id,
                "student_id": student_id,
                "date": date,
                "status": status_value
            })

            insert_attendance_query = """
                INSERT INTO attendance (
                    attendance_id, student_id, date, status
                ) VALUES (%s, %s, %s, %s)
            """

            # Create a dictionary for the attendance data
            attendance_data = {
                "attendance_id": attendance_id,
                "student_id": student_id,
                "date": date,
                "status": status_value
            }

            # Extract values from the dictionary
            attendance_values = tuple(attendance_data.values())

            # Insert attendance data into the database
            try:
                with connection.cursor() as cursor:
                    cursor.execute(insert_attendance_query, attendance_values)
                    connection.commit()
                    st.success("Attendance Record Submitted and Saved to Database Successfully!")
            except Exception as e:
                st.error(f"An error occurred: {e}")


            st.success("Attendance Record Submitted Successfully!")
            attendance_df = pd.DataFrame(attendance_data)
            st.dataframe(attendance_df)

            

def collect_academic_info():
    st.title("Academic Records Collection")
    academic_data = []

    with st.form("academic_form", clear_on_submit=True):
        student_id = st.text_input("Student ID")
        subject_id = st.text_input("Subject ID")
        attendance_percentage = st.slider("Attendance Percentage", 0.0, 100.0, step=0.1)
        assignment_score = st.slider("Assignment Score", 0.0, 100.0, step=0.1)
        exam_score = st.slider("Exam Score", 0.0, 100.0, step=0.1)

        submitted = st.form_submit_button("Submit")
        if submitted:
            final_grade = round((attendance_percentage * 0.2) + 
                                (assignment_score * 0.2) + 
                                (exam_score * 0.6), 2)
            academic_data.append({
                "student_id": student_id,
                "subject_id": subject_id,
                "attendance_percentage": attendance_percentage,
                "assignment_score": assignment_score,
                "exam_score": exam_score,
                "final_grade": final_grade
            })
            st.success("Academic Record Submitted Successfully!")
            academic_df = pd.DataFrame(academic_data)
            st.dataframe(academic_df)

def collect_teacher_evaluation():
    st.title("Teacher Evaluation Collection")
    evaluation_data = []
    comments_mapping = {
        'Excellent performance': 10,
        'Good performance': 8,
        'Fair performance': 6,
        'Average performance': 4,
        'Below average performance': 2,
        'Very bad performance': 0
    }

    with st.form("evaluation_form", clear_on_submit=True):
        student_id = st.text_input("Student ID")
        subject_id = st.text_input("Subject ID")
        evaluation_date = st.date_input("Evaluation Date")
        comment = st.selectbox("Comment", list(comments_mapping.keys()))

        submitted = st.form_submit_button("Submit")
        if submitted:
            point = comments_mapping[comment]
            evaluation_data.append({
                "student_id": student_id,
                "subject_id": subject_id,
                "evaluation_date": evaluation_date,
                "comments": comment,
                "point": point
            })
            st.success("Teacher Evaluation Submitted Successfully!")
            evaluation_df = pd.DataFrame(evaluation_data)
            st.dataframe(evaluation_df)

def collect_class_assignment():
    st.title("Class-Subject-Teacher Assignment Collection")
    assignment_data = []

    with st.form("class_assignment_form", clear_on_submit=True):
        classes = ['Jss 1', 'Jss 2', 'Jss 3', 'SS1', 'SS2', 'SS3']

        class_name = st.selectbox("Class Name", classes )
        subject_id = st.text_input("Subject ID")
        teacher_id = st.text_input("Teacher ID")

        submitted = st.form_submit_button("Submit")
        if submitted:
            assignment_data.append({
                "class": class_name,
                "subject_id": subject_id,
                "teacher_id": teacher_id
            })
            st.success("Class Assignment Submitted Successfully!")
            assignment_df = pd.DataFrame(assignment_data)
            st.dataframe(assignment_df)