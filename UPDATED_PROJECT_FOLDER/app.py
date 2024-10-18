import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import os
import pyodbc
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder
from collecting_data_from_db import getting

current_dir = os.path.dirname(os.path.abspath(__file__))
encoder_path = os.path.join(current_dir,'encoders')
model_path = os.path.join(current_dir,'models')
img_path = os.path.join(current_dir,'picture.png')
img_path1 = os.path.join(current_dir,'picture2.png')
img_path2 = os.path.join(current_dir,'picture3.png')
dataset_dir = os.path.join(current_dir, 'datasets')
# List of columns that have encoders
encoded_columns = ['class', 'gender', 'ethnicity', 'family_size', 'favorite_subjects', 'sleep', 'average_study_time', 'distance_to_school', 'mode_of_transportation', 'gender_parent', 'familyincomerange', 'numberofchildren', 'occupation', 'houseownership', 
'educationlevel', 'maritalstatus', 'location', 'age_parent', 'employmentstatus', 'transportationmode', 'subject']

# Dictionary to store the loaded encoders
encoders = {}

# Load all encoders
for col in encoded_columns:
    with open(rf'{encoder_path}\{col}_encoder.pkl', 'rb') as f:
        encoders[col] = pickle.load(f)

def load_scaler():
    img_path = os.path.join(current_dir,'picture.png')
    with open(rf'{encoder_path}\scaler.pkl', 'rb') as f:
        return pickle.load(f)

def load_model():
    with open(rf'{model_path}\best_model.pkl', 'rb') as f:
        return pickle.load(f)

# Prediction evaluation function
def evaluate_prediction(prediction):
    if prediction > 0.75:
        return "Student has a very high chance of passing exam", 'green'
    elif 0.5 < prediction <= 0.75:
        return "Student might surely get above average ", 'blue'
    elif 0.25 < prediction <= 0.5:
        return "Student might surely get below average", 'orange'
    else:
        return "Student has a very high of not doing well in the forthcoming exam", 'red'

# Prediction function
def predict_student_performance(features, model, scaler):
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    prediction_proba = model.predict_proba(features_scaled)[0][1]  # Assuming binary classification
    recommend, color = evaluate_prediction(prediction_proba)
    return recommend, color, prediction_proba, prediction

# Fetch student details from the database
@st.cache_data
def get_student_details(student_id_value, subject_value):
    file_path = os.path.join(dataset_dir, f'school_dataset.csv')
    df  = pd.read_csv(file_path)
    df = df.query("student_id == @student_id_value and subject == @subject_value").tail(1)
    # df.query("student_id = student_ids ")
    # query = """
    #     SELECT TOP 5 * FROM school_dataset 
    #     WHERE student_id = ? AND subject = ?
    #     """
    # df = pd.read_sql_query(query, cnxn, params=[student_id, subject])
    
    if df.empty:
        st.error("No records found for this Student ID and subject.")
        return None
    
    return df



# Streamlit App UI
def main():

    st.set_page_config(
    layout="wide",  # Optional: Makes the page layout wider
    # initial_sidebar_state="collapsed",
    page_title="My Streamlit App")
    # st.sidebar.header("Oginigba Comprehensive Secondary School Academic Prediction/Information")
    # st.sidebar.markdown("<h1 style='font-size: 30px;'>Oginigba Comprehensive Secondary School Academic Prediction/Information</h1>", unsafe_allow_html=True)
    st.sidebar.image(img_path , use_column_width=True)

    # Sidebar: User Selection
    section = st.sidebar.radio("Select use", ['Home','Performance Prediction', 'Data Collection', 'Mark Attendance', 'Visualizations', 'Descriptive Statistics'])
    if section == 'Home':
        st.markdown("<h1 style='font-size: 35px;'><center>🎓 Welcome to Oginigba Student Performance Predictor!</center></h1>", unsafe_allow_html=True)


        # List of images for the slideshow
        images = [
            img_path,
            img_path1,
            img_path2
        ]
        
        # Create a placeholder for the image
        image_placeholder = st.empty()
        count = 0
        # Slideshow functionality
        while count < 4:
            for image in images:
                # Update the placeholder with the current image
                image_placeholder.image(image, use_column_width=True)
                time.sleep(2)  # Change slide every 3 seconds
                count += 1
                

        # Introduction
        st.markdown(
            """
            ## 🌟 Transforming Education through Data Science  
            Welcome to the **Oginigba Student Performance Predictor**, a cutting-edge platform designed to reshape how we monitor and enhance academic success. With the power of **machine learning** and **advanced analytics**, we provide schools, teachers, and administrators with valuable insights to predict student outcomes, track attendance, and unlock each learner's full potential.

            ### 💼 Who Can Benefit:
            - **School Administrators:** Get an overview of student performance trends to guide decision-making.
            - **Teachers:** Monitor students’ progress and identify those needing extra support.
            - **Parents:** Understand how their children are performing and encourage active participation.
            - **Students:** Track their own progress.

            ---
            ## 🛠️ Features at a Glance:
            - **🎯 Performance Prediction:** Predict individual student outcomes to help plan interventions early.
            - **📋 Data Collection:** Manage academic records, attendance logs, and behavioral reports in one place.
            - **📅 Attendance Tracker:** Ensure class participation is consistent and highlight absenteeism trends.
            - **📊 Visual Analytics:** Generate real-time graphs and dashboards to analyze academic performance.
            - **📈 Descriptive Statistics:** Summarize key statistics such as average scores, attendance percentages, and trends over time.
            - **🏫 Multi-Level Analysis:** Track performance across subjects, classrooms, and grade levels.

            ---
            ## 🔍 How to Use the App:
            1. **Home:** Begin your journey from here with a warm welcome and guidance on app features.
            2. **Performance Prediction:** Upload students’ past records to forecast their future performance.
            3. **Data Collection:** Use our easy-to-fill forms to gather and manage student data.
            4. **Attendance Tracker:** Log student attendance and track it visually.
            5. **Visualizations:** Create interactive graphs to analyze performance trends and attendance patterns.
            6. **Descriptive Statistics:** View summaries of student performance metrics to make data-driven decisions.

            ---
            ## 📢 Why Oginigba Predictor Stands Out:
            - **⚡ Early Interventions:** Identify students at risk of falling behind and act proactively.
            - **📚 Support Personalized Learning:** Tailor teaching strategies based on performance insights.
            - **🌐 Seamless Data Handling:** Manage large datasets with ease and export insights in multiple formats.
            - **📱 Easy Access:** Use the platform anytime, anywhere on both desktop and mobile devices.
            - **🚀 Future-Proof:** Stay ahead with continuous updates and evolving prediction algorithms.

            ---
            ## 🚀 Impact of the Platform:
            - **Student Success Stories:** Empower students to reach their full potential by understanding their strengths and weaknesses.
            - **Community Growth:** Engage parents, teachers, and administrators to work together for the students’ success.
            - **Improved Attendance:** Foster a culture of accountability with easy attendance tracking and visual alerts.
            - **Optimized Teaching Strategies:** Use predictive insights to refine lesson plans and improve teaching outcomes.

            ---
            ## 🌱 Let’s Get Started!
            Explore the app using the **sidebar on the left**. Each section is designed to guide you step-by-step in managing and analyzing student data. Whether you're predicting performance, tracking attendance, or generating visual reports, we’ve got you covered!

            ---
            ### 🔧 Need Help? 
            Feel free to reach out if you have any questions or need technical assistance. Together, we can shape a brighter future for every student at Oginigba!

            **“Data is the new oil.”** – Use it wisely, and you can fuel success!
            """
        )

    elif section == 'Data Collection':
        st.header('Data Collection')
        st.write('Click on the necessary button to upload the information')
        from collection_script import collecting_student_info,collecting_parent_info,collecting_teacher_info,collect_subject_info,collect_attendance_info,collect_academic_info,collect_teacher_evaluation,collect_class_assignment
        if st.button('Student Data Collection'):
            collecting_student_info()
            time.sleep(5)
            # st.success('Successfully added!')
        elif st.button('Parent Data Collection'):
            collecting_parent_info()
        elif st.button('Teacher Data Collection'):
            collecting_teacher_info()

        elif st.button("Subject Information"):
            collect_subject_info()
        elif st.button("Attendance Record"):
            collect_attendance_info()
        elif st.button("Academic Records"):
            collect_academic_info()
        elif st.button("Teacher Evaluation"):
            collect_teacher_evaluation()
        elif st.button("Class Assignment"):
            collect_class_assignment()

            

    elif section == 'Performance Prediction':
        st.markdown("<h1 style='font-size: 35px;'><center>Oginigba Comprehensive Secondary School Performance Prediction/Information</center></h1>", unsafe_allow_html=True)

        st.subheader("Checking Student's information for prediction...")

        # Input fields for Student ID and Subject
        student_id = st.text_input("Enter the Student ID e.g OGB/2024/18", help="Enter a valid Student ID")
        subject = st.text_input("Enter the Subject e.g Basic Technology", help="Enter the subject for the prediction")


        if student_id and subject:
            student_info = get_student_details(student_id, subject)
            
            if student_info is not None:
                st.success("Student details found!")

                
                student_id = student_info['student_id'].values[0]
                student_name = student_info['student_name'].values[0]
                student_class = student_info['class'].values[0]
                age = student_info['age'].values[0]
                gender = student_info['gender'].values[0]
                ethnicity = student_info['ethnicity'].values[0]
                disability_status = student_info['disability_status'].values[0]
                family_size = student_info['family_size'].values[0]
                favorite_subjects = student_info['favorite_subjects'].values[0]
                access_to_constant_electricity = student_info['access_to_constant_electricity'].values[0]
                on_scholarship = student_info['on_scholarship'].values[0]
                sleep = student_info['sleep'].values[0]
                average_study_time = student_info['average_study_time'].values[0]
                enjoy_reading = student_info['enjoy_reading'].values[0]
                enjoy_dancing = student_info['enjoy_dancing'].values[0]
                enjoy_socialising = student_info['enjoy_socialising'].values[0]
                anxiety_before_during_exams = student_info['anxiety_before_during_exams'].values[0]
                distance_to_school = student_info['distance_to_school'].values[0]
                mode_of_transportation = student_info['mode_of_transportation'].values[0]

                # Parent Information
                # parent_id = student_info['parent_id'].values[0]
                parent_name = student_info['parent_name'].values[0]
                gender_parent = student_info['gender_parent'].values[0]
                email = student_info['email'].values[0]
                family_income_range = student_info['familyincomerange'].values[0]
                number_of_children = student_info['numberofchildren'].values[0]
                occupation = student_info['occupation'].values[0]
                house_ownership = student_info['houseownership'].values[0]
                education_level = student_info['educationlevel'].values[0]
                marital_status = student_info['maritalstatus'].values[0]
                location = student_info['location'].values[0]
                age_parent = student_info['age_parent'].values[0]
                work_hours_per_week = student_info['workhoursperweek'].values[0]
                employment_status = student_info['employmentstatus'].values[0]
                parent_transportation_mode = student_info['transportationmode'].values[0]
                internet_access = student_info['internetaccess'].values[0]
                # qualification = student_info['qualification'].values[0]
                # parent_phone_number = student_info['phone_number'].values[0]

                # Academic Information
                # subject_id = student_info['subject_id'].values[0]
                attendance_percentage = student_info['attendance_percentage'].values[0]
                assignment_score = student_info['assignment_score'].values[0]
                exam_score = student_info['exam_score'].values[0]
                final_grade = student_info['final_grade'].values[0]
                # academic_record_id = student_info['academic_record_id'].values[0]
                # evaluation_date = student_info['evaluation_date'].values[0]
                # comments = student_info['comments'].values[0]
                point = student_info['point'].values[0]
                subject = student_info['subject'].values[0]
                totalDaysPresent = student_info['totaldayspresent'].values[0]
                attendancePercentage_overall = student_info['attendancepercentage_overall'].values[0]

                # Target (assuming this is the label or output variable)
                # target = student_info['target'].values[0]


                # Display current teacher and parent information
                st.write(f"Student ID : {student_id}")
                st.write(f"Student Name : {student_name}")
                st.write(f"Parent Name: {parent_name}")
                # st.write(f"Parent Phone: {parent_phone_number}")
                st.write(f"Parent email: {email}")

                st.write('----------------------------------------------------------')

                st.write('Student Information :')
                # Create selectboxes with the available options for each column
                student_class = st.selectbox("Class", encoders['class'].classes_, 
                                            index=encoders['class'].classes_.tolist().index(student_class))
                student_class = encoders['class'].transform([student_class])[0]

                age = st.slider("Age",9,20, value=age)  # Numeric input for age

                gender = st.selectbox("Gender", encoders['gender'].classes_, 
                                    index=encoders['gender'].classes_.tolist().index(gender))
                gender = encoders['gender'].transform([gender])[0]

                subject = st.selectbox("Subject", encoders['subject'].classes_, 
                                    index=encoders['subject'].classes_.tolist().index(subject))
                subject = encoders['subject'].transform([subject])[0]

                ethnicity = st.selectbox("Ethnicity", encoders['ethnicity'].classes_, 
                                        index=encoders['ethnicity'].classes_.tolist().index(ethnicity))
                ethnicity = encoders['ethnicity'].transform([ethnicity])[0]

                family_size = st.selectbox("Family Size", encoders['family_size'].classes_, 
                                        index=encoders['family_size'].classes_.tolist().index(family_size))
                family_size = encoders['family_size'].transform([family_size])[0]

                favorite_subjects = st.selectbox("Favorite Subjects", encoders['favorite_subjects'].classes_, 
                                                index=encoders['favorite_subjects'].classes_.tolist().index(favorite_subjects))
                favorite_subjects = encoders['favorite_subjects'].transform([favorite_subjects])[0]

                sleep = st.selectbox("Sleep", encoders['sleep'].classes_, 
                                    index=encoders['sleep'].classes_.tolist().index(sleep))
                sleep = encoders['sleep'].transform([sleep])[0]

                average_study_time = st.selectbox("Average Study Time", encoders['average_study_time'].classes_, 
                                                index=encoders['average_study_time'].classes_.tolist().index(average_study_time))
                average_study_time = encoders['average_study_time'].transform([average_study_time])[0]

                distance_to_school = st.selectbox("Distance to School", encoders['distance_to_school'].classes_, 
                                                index=encoders['distance_to_school'].classes_.tolist().index(distance_to_school))
                distance_to_school = encoders['distance_to_school'].transform([distance_to_school])[0]

                mode_of_transportation = st.selectbox("Mode of Transportation", encoders['mode_of_transportation'].classes_, 
                                                    index=encoders['mode_of_transportation'].classes_.tolist().index(mode_of_transportation))
                mode_of_transportation = encoders['mode_of_transportation'].transform([mode_of_transportation])[0]

                disability_status = st.radio("Disability_Status", [True, False], index=[True, False].index(disability_status))  # Disability status
                access_to_constant_electricity = st.radio("Access_to_Constant_Electricity", [True, False], index=[True, False].index(access_to_constant_electricity))  # Access to electricity
                on_scholarship = st.radio("On_Scholarship", [True, False], index=[True, False].index(on_scholarship))  # On scholarship
                enjoy_reading = st.radio("Enjoy_Reading", [True, False], index=[True, False].index(enjoy_reading))  # Enjoy reading
                enjoy_dancing = st.radio("Enjoy_Dancing", [True, False], index=[True, False].index(enjoy_dancing))  # Enjoy dancing
                enjoy_socialising = st.radio("Enjoy_Socialising", [True, False], index=[True, False].index(enjoy_socialising))  # Enjoy socialising
                anxiety_before_during_exams = st.radio("Anxiety_Before_During_Exams", [True, False], index=[True, False].index(anxiety_before_during_exams))  # Anxiety before/during exams
                
                st.write('-------------------------------------------------')
                st.write("Parent's Information" )
                gender_parent = st.selectbox("Parent's Gender", encoders['gender_parent'].classes_, 
                                            index=encoders['gender_parent'].classes_.tolist().index(gender_parent))
                gender_parent = encoders['gender_parent'].transform([gender_parent])[0]


                family_income_range = st.selectbox("Family Income Range", encoders['familyincomerange'].classes_, 
                                                index=encoders['familyincomerange'].classes_.tolist().index(family_income_range))
                family_income_range = encoders['familyincomerange'].transform([family_income_range])[0]

                number_of_children = st.selectbox("Number of Children", encoders['numberofchildren'].classes_, 
                                                index=encoders['numberofchildren'].classes_.tolist().index(number_of_children))
                number_of_children = encoders['numberofchildren'].transform([number_of_children])[0]

                occupation = st.selectbox("Occupation", encoders['occupation'].classes_, 
                                        index=encoders['occupation'].classes_.tolist().index(occupation))
                occupation = encoders['occupation'].transform([occupation])[0]

                house_ownership = st.selectbox("House Ownership", encoders['houseownership'].classes_, 
                                            index=encoders['houseownership'].classes_.tolist().index(house_ownership))
                house_ownership = encoders['houseownership'].transform([house_ownership])[0]

                education_level = st.selectbox("Education Level", encoders['educationlevel'].classes_, 
                                            index=encoders['educationlevel'].classes_.tolist().index(education_level))
                education_level = encoders['educationlevel'].transform([education_level])[0]

                marital_status = st.selectbox("Marital Status", encoders['maritalstatus'].classes_, 
                                            index=encoders['maritalstatus'].classes_.tolist().index(marital_status))
                marital_status = encoders['maritalstatus'].transform([marital_status])[0]

                location = st.selectbox("Location", encoders['location'].classes_, 
                                        index=encoders['location'].classes_.tolist().index(location))
                location = encoders['location'].transform([location])[0]

                age_parent = st.selectbox("Parent's Age", encoders['age_parent'].classes_, 
                                        index=encoders['age_parent'].classes_.tolist().index(age_parent))
                age_parent = encoders['age_parent'].transform([age_parent])[0]

                employment_status = st.selectbox("Employment Status", encoders['employmentstatus'].classes_, 
                                                index=encoders['employmentstatus'].classes_.tolist().index(employment_status))
                employment_status = encoders['employmentstatus'].transform([employment_status])[0]

                work_hours_per_week = st.slider("Work Hours Per Week",0.0,60.0, value=work_hours_per_week,step = 0.01)  # Work hours per week
                
                internet_access = st.radio("Internet Access", [True, False], index=[True, False].index(internet_access))  # Anxiety before/during exams
                
                parent_transportation_mode = st.selectbox("Transportation Mode", encoders['transportationmode'].classes_, 
                                                index=encoders['transportationmode'].classes_.tolist().index(parent_transportation_mode))

                parent_transportation_mode = encoders['transportationmode'].transform([parent_transportation_mode])[0]

                st.write('-------------------------------------------------')
                
               ## Numeric fields
                st.write('Performance evaluation metrics :')
                attendance_percentage = st.slider("Attendance Percentage",0.0,100.0, value=attendance_percentage)  # Attendance percentage
                assignment_score = st.slider("Assignment Score",0.0,100.0, value=assignment_score)  # Assignment score
                exam_score = st.slider("Exam Score", 0.0,100.0,value=exam_score)  # Exam score
                final_grade = st.slider("Final Grade",0.0,100.0, value=final_grade)  # Final grade
                point = st.number_input("Point",0,10 ,value=point)  # Point
                totalDaysPresent = st.number_input("Total Days Present", value=totalDaysPresent)  # Total days present
                attendancePercentage_overall = st.slider("Attendance Percentage (Overall)",0.0,100.0, value=attendancePercentage_overall)  # Attendance percentage overall


                # Gather inputs into a dictionary
                features = {
                    'class': [student_class],
                    'age': [age],
                    'gender': [gender],
                    'ethnicity': [ethnicity],
                    'disability_status': [disability_status],
                    'family_size': [family_size],
                    'favorite_subjects': [favorite_subjects],
                    'access_to_constant_electricity': [access_to_constant_electricity],
                    'on_scholarship': [on_scholarship],
                    'sleep': [sleep],
                    'average_study_time': [average_study_time],
                    'enjoy_reading': [enjoy_reading],
                    'enjoy_dancing': [enjoy_dancing],
                    'enjoy_socialising': [enjoy_socialising],
                    'anxiety_before_during_exams': [anxiety_before_during_exams],
                    'distance_to_school': [distance_to_school],
                    'mode_of_transportation': [mode_of_transportation],
                    'gender_parent': [gender_parent],
                    'familyincomerange': [family_income_range],
                    'numberofchildren': [number_of_children],
                    'occupation': [occupation],
                    'houseownership': [house_ownership],
                    'educationlevel': [education_level],
                    'maritalstatus': [marital_status],
                    'location': [location],
                    'age_parent': [age_parent],
                    'workhoursperweek': [work_hours_per_week],
                    'employmentstatus': [employment_status],
                    'transportationmode': [parent_transportation_mode],
                    'internetaccess': [internet_access],
                    'attendance_percentage': [attendance_percentage],
                    'assignment_score': [assignment_score],
                    'exam_score': [exam_score],
                    'final_grade': [final_grade],
                    'point': [point],
                    'subject': [subject],
                    'totaldayspresent': [totalDaysPresent],
                    'attendancepercentage_overall': [attendancePercentage_overall],
                }

                # Convert dictionary to DataFrame
                features_df = pd.DataFrame(features)

                
                # Prediction Button
                if st.button("Predict"):
                    
                    scaler = load_scaler()
                    model = load_model()

                    # scaled_data = scaler.transform(features_df)
                    # prediction = model.predict(scaled_data)
    #                 # Perform prediction
                    recommendation, color, probability, result = predict_student_performance(features_df, model, scaler)
                    
                    prediction_text = f"Model Prediction: {result[0]}"
                    prediction_proba = f"Prediction Probability: {probability:.2f}"
                    st.markdown(f"<h4 style='font-size: 24px;'>{prediction_text}</h4>", unsafe_allow_html=True)

                    st.markdown(f"<h4 style='color: {color};'>{recommendation}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<h4 style='font-size: 24px;'>{prediction_proba}</h4>", unsafe_allow_html=True)

    elif section == 'Mark Attendance':
        # Get the current directory of the script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        school_dataset = os.path.join(dataset_dir,'school_dataset.csv')
        attendance_dataset = os.path.join(dataset_dir,'attendance_data.csv')
        barcode_folder = os.path.join(current_dir, 'barcodes')
        os.makedirs(barcode_folder, exist_ok=True)

        df = pd.read_csv(school_dataset)
        attendance_df = pd.read_csv(attendance_dataset)

        from mark_att import generate_barcode,log_attendance,scan_barcode

        # Streamlit interface
        st.title("Student Attendance Marking System with Barcode Scanner")
        #password = 9090
        # Generate barcodes for all students
        if st.button("Generate Barcodes for all Students"):
            # ad_password = st.text_input('Enter Admin Password to Generate Barcodes',type='password', max_chars = 4)
            # if ad_password == 9090:
            #     # Progress bar
            progress_bar = st.progress(0)  # Initialize progress bar
            total_students = len(df['student_id'].unique())

            for idx, student_id in enumerate(df['student_id'].unique()):
                id = student_id.replace('/', '_')  # Handle special characters in student ID
                generate_barcode(id)  # Call the barcode generation function

                # Update progress bar
                progress = (idx + 1) / total_students  # Calculate progress
                progress_bar.progress(progress)  # Update progress bar

            st.success("All barcodes generated!")
            st.balloons()  # Show success balloons
            # elif ad_password == '****':
            #     st.write('waiting.....')
            # else:
            #     st.write('Incorrect Password, Please contact the administrator(Arrow Team)')
        # Button to start scanning
        elif st.button("Scan Barcode"):
            st.text("Please scan the barcode...")
            student_id = scan_barcode()  # Start scanning when button is pressed
            st.success(f"Recorded attendance for {student_id} at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.")
           

    elif section == 'Visualizations':
        st.markdown('# Real Time Visualizations')
        from plottings import to_plot
        to_plot()

    elif section == 'Descriptive Statistics':
        from description import describing
        describing()


if __name__ == '__main__':
    getting()
    main()
