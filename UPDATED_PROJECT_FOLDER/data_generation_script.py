import pandas as pd
import numpy as np
import random
import datetime as dtt
import os
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, 'datasets')
os.makedirs(dataset_dir, exist_ok=True)

# Setting seed 
random.seed(42)
np.random.seed(42)

#initializing important variables
no_of_students = 500
no_of_parents = 500
no_of_teachers = 25
no_of_attendance_records = 70

# List of Nigerian first names and last names
first_names = [
    'Ada', 'Chinedu', 'Olufemi', 'Amina', 'Uche', 'Damilola', 'Nkechi',
    'Sadiq', 'Temitope', 'Kelechi', 'Ayo', 'Funke', 'Tunde', 'Ifeoma',
    'Chika', 'Bukola', 'Nneka', 'Yemi', 'Ijeoma', 'Chukwuemeka', 'Olamide',
    'Tobi', 'Oluwaseun', 'Nnamdi', 'Zainab', 'Chinonso', 'Ifeanyichukwu', 
    'Obinna', 'Nkemdilim', 'Ugo', 'Chidera', 'Duru', 'Obiageli', 'Ekene', 
    'Jumoke', 'Adebisi', 'Aderonke', 'Opeyemi', 'Fisayo', 'Ihuoma', 
    'Chinwe', 'Ujunwa', 'Ngozi', 'Doyin', 'Fola', 'Bola', 'Kehinde',
    'Ololade', 'Chinyere', 'Bimbo', 'Temiloluwa', 'Kanyinsola', 'Opeyemi'
]

last_names = [
    'Okeke', 'Adeyemi', 'Adebayo', 'Ibrahim', 'Nwosu', 'Obi', 'Ajayi',
    'Abiola', 'Ojo', 'Hassan', 'Eze', 'Johnson', 'Suleiman', 'Uba',
    'Onyekachi', 'Dada', 'Balogun', 'Abubakar', 'Adeola', 'Chukwuma',
    'Ogunleye', 'Nwankwo', 'Olaniyi', 'Idowu', 'Adesina', 'Ogunbiyi',
    'Okafor', 'Ezeani', 'Kalu', 'Nduka', 'Okwudili', 'Obinna', 'Nwodo', 
    'Ogundipe', 'Tajudeen', 'Obadiah', 'Awojobi', 'Ogunyemi', 'Akinyemi',
    'Olatunde', 'Fadeyi', 'Akanbi', 'Ajibola', 'Olowu', 'Ojo', 'Aiyegbusi'
]

# New list of Nigerian first names and last names
teacher_first_names = [
    'Abayomi', 'Omolara', 'Chigozie', 'Folake', 'Chinaza', 'Gbenga', 'Zainab', 
    'Abdul', 'Funmilayo', 'Omotola', 'Chukwuemeka', 'Adewale', 'Fatima', 
    'Osaretin', 'Benedicta', 'Adebisi', 'Chioma', 'Emeka', 'Tijani', 'Ngozi', 
    'Bolaji', 'Ezinne', 'Oluwadamilola', 'Olumide', 'Suleiman', 'Aisha', 
    'Adeola', 'Ikemefuna', 'Bukola', 'Seun', 'Nkeiru', 'Ifedayo', 'Modupe', 
    'Titi', 'Babajide', 'Abisola', 'Oluwafemi', 'Ndidi', 'Amaka', 'Tunde', 
    'Aminat', 'Seyi', 'Olufunke', 'Ibrahim', 'Olusegun', 'Adetokunbo', 
    'Solomon', 'Kunle', 'Chinedu', 'Bose', 'Sade', 'Latifat'
]

teacher_last_names = [
    'Ademola', 'Ogunmola', 'Adegoke', 'Okechukwu', 'Olawale', 'Abiola', 
    'Adebanjo', 'Ezeani', 'Okoro', 'Ojo', 'Udo', 'Onifade', 'Afolabi', 
    'Akinbami', 'Adebayo', 'Ogbonna', 'Ogunleye', 'Udeh', 'Ibrahim', 
    'Nwachukwu', 'Balogun', 'Umeh', 'Abubakar', 'Adegbola', 'Anyanwu', 
    'Oyinlola', 'Okafor', 'Ajayi', 'Sowande', 'Adelakun', 'Chukwuma', 
    'Omotayo', 'Alabi', 'Olaniyan', 'Adeyemi', 'Adedoyin', 'Awolowo', 
    'Akintola', 'Yusuf', 'Adefemi', 'Adewumi', 'Olowu', 'Fasuyi', 
    'Abayomi', 'Ogunjimi', 'Obi', 'Kareem', 'Odusanya', 'Adebowale', 
    'Sanni', 'Okeowo'
]


# Generating unique combinations of first and last names
unique_names = set()

while len(unique_names) < no_of_students:
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    full_name = f"{first_name} {last_name}"
    unique_names.add(full_name)

# Converting set to a list
student_names = list(unique_names)

# Generating unique combinations of first and last names for teachers
unique_names = set()

while len(unique_names) < no_of_teachers:
    first_name = random.choice(teacher_first_names)
    last_name = random.choice(teacher_last_names)
    full_name = f"{first_name} {last_name}"
    unique_names.add(full_name)

# Converting set to a list
teacher_names = list(unique_names)

# Assigning titles based on gender (Mr. for male and Mrs. for female)
gender = [random.choice(['Male', 'Female']) for _ in range(no_of_teachers)]
teacher_names_with_titles = [
    f"Mr. {name}" if gender[i] == 'Male' else f"Mrs. {name}" 
    for i, name in enumerate(teacher_names)
]


subjects = ['Mathematics', 'English Language', 'Chemistry', 'Biology', 
            'Physics', 'Geography', 'History', 'Civic Education', 
            'Agricultural Science', 'Computer Studies', 'Business Studies', 
            'Fine Arts'] 


def generate_age(class_type):
    age_ranges = {
        'Jss 1': (9, 12),
        'Jss 2': (10, 13),
        'Jss 3': (11, 13),
        'SS1': (13, 15),
        'SS2': (14, 16),
        'SS3': (14, 18)
    }
    min_age, max_age = age_ranges[class_type]
    return random.randint(min_age, max_age)

student_id = [f'OGB/2024/{str(i)}' for i in range(no_of_students)]
class_type = random.choices(['Jss 1', 'Jss 2', 'Jss 3', 'SS1', 'SS2', 'SS3'], 
                            weights=[0.1, 0.2, 0.2, 0.1, 0.1, 0.3], 
                            k=no_of_students)
ages = [generate_age(ct) for ct in class_type]
gender = random.choices(['Male', 'Female'], weights=[0.4, 0.6], k=no_of_students)
ethnicity = [random.choice(["Hausa", "Igbo", "Yoruba"]) for _ in range(no_of_students)]
disability_status = random.choices([True, False], weights=[0.9, 0.1], k=no_of_students)
family_size = random.choices(['0', '1', '2', '3', '4', 'more than 5'], 
                             weights=[0.1, 0.3, 0.25, 0.15, 0.15, 0.05], 
                             k=no_of_students)
favorite_subjects = [random.choice(subjects) for _ in range(no_of_students)]
access_to_constant_electricity = random.choices([True, False], weights=[0.9, 0.1], k=no_of_students)
on_scholarship = random.choices([True, False], weights=[0.9, 0.1], k=no_of_students)
study = [random.choice(['<1 hour', '1 - 2 hours', '2 - 3 hours', '3 - 4 hours', '4 - 6 hours', '> 6 hours']) for _ in range(no_of_students)]
sleep = [random.choice(['<6 hours', '6 - 8 hours', '>8 hours']) for _ in range(no_of_students)]
anxiety = [random.choice([True, False]) for _ in range(no_of_students)]
reading = [random.choice([True, False]) for _ in range(no_of_students)]
dancing = [random.choice([True, False]) for _ in range(no_of_students)]
socialising = [random.choice([True, False]) for _ in range(no_of_students)]
school_distance = [random.choice(['very close', 'close', 'far', 'very far']) for _ in range(no_of_students)]
mode_of_transportation = [random.choice(['trekking', 'bicycle', 'tricycle', 'school bus', 'private/family car']) for _ in range(no_of_students)]

# Creating a DataFrame
students_df = pd.DataFrame({
    'student_id': student_id,
    'student_name' : student_names,
    'class': class_type,
    'age': ages,
    'gender': gender,
    'ethnicity': ethnicity,
    'disability_status': disability_status,
    'family_size': family_size,
    'favorite_subjects': favorite_subjects,
    'access_to_constant_electricity': access_to_constant_electricity,
    'on_scholarship': on_scholarship,
    'sleep': sleep,
    'average_study_time': study,
    'enjoy_reading': reading,
    'enjoy_dancing': dancing,
    'enjoy_socialising': socialising,
    'anxiety_before_during_exams': anxiety,
    'distance_to_school': school_distance,
    'mode_of_transportation': mode_of_transportation
})



#Parents/Guardian information
#important here are family income range, number of children, occupation, 
parent_id = [f'parent#{str(i+1)}' for i in range(no_of_parents)]  # Generate parent IDs
student_id_parent = student_id[:no_of_parents]
parent_name = [f'{random.choice(first_names)} {name.split(' ')[-1]}' for name in student_names]
email = [f'{i.split(' ')[-2]}{i.split(' ')[-1]}@gmail.com'.lower() for i in parent_name]
gender = [random.choice(['Male', 'Female']) for _ in range(no_of_parents)]
family_income_range = [random.choice(['<#30,000', '#30,001 - #70,000', '#70, 001 - #120,000', '#120, 001 - #200,000','>#200,001']) for _ in range(no_of_parents)]
number_of_children = [random.choice(['0', '1', '2', '3', '4', '5','more than 5']) for _ in range(no_of_parents)]
occupation = [random.choice(['Unemployed', 'Self Employed', 'Employed']) for _ in range(no_of_parents)]
house_ownership = [random.choice(['Rent','Owned']) for _ in range(no_of_parents)]
education_level = [random.choice(['High School', 'Associate Degree', 'Bachelor\'s Degree', 'Master\'s Degree', 'PhD']) for _ in range(no_of_parents)]
marital_status = [random.choice(['Married', 'Single', 'Divorced', 'Widowed']) for _ in range(no_of_parents)]
location = [random.choice(['Urban', 'Suburban', 'Rural']) for _ in range(no_of_parents)]
age_range = [random.choice(['25-34', '35-44', '45-54', '55-64', 'older']) for _ in range(no_of_parents)]
work_hours_per_week = [random.randint(0, 60) for _ in range(no_of_parents)]  # Hours worked per week
employment_status = [random.choice(['Full-time', 'Part-time', 'Freelance']) for _ in range(no_of_parents)]
transportation_mode = [random.choice(['Car', 'Bus', 'Bike', 'Trekking']) for _ in range(no_of_parents)]
internet_access = [random.choice([True, False]) for _ in range(no_of_parents)]
qualification = [random.choice(['Bachelors', 'Masters', 'PhD', 'Diploma','None']) for _ in range(no_of_parents)]
phone_number = [f'+234{random.randint(1000000000, 9999999999)}' for _ in range(no_of_parents)]

parent_df = pd.DataFrame({
    'parent_id': parent_id,
    'student_id':student_id_parent,
    'parent_name' : parent_name,
    'gender' : gender,
    'email' : email,
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
    'qualification':qualification,
    'phone_number' : phone_number
})


# Employee information
teacher_id = [f'OGBTeacher/2024/{str(i)}' for i in range(no_of_teachers)]
salary_range = [random.choice(['<#50,000', '#50,001 - #100,000', '#100,000 - #150,000', '#150,000+']) for _ in range(no_of_teachers)]
department = [random.choice(subjects) for _ in range(no_of_teachers)]
qualification = [random.choice(['Bachelors', 'Masters', 'PhD', 'Diploma']) for _ in range(no_of_teachers)]
employment_status = [random.choice(['Full-time', 'Part-time']) for _ in range(no_of_teachers)]
gender = [random.choice(['Male', 'Female']) for _ in range(no_of_teachers)]
years_of_experience = [random.randint(1, 30) for _ in range(no_of_teachers)]
phone_number = [f'+234{random.randint(1000000000, 9999999999)}' for _ in range(no_of_teachers)]
email = [f'{i.split(' ')[-2]}{i.split(' ')[-1]}@gmail.com' for i in teacher_names_with_titles]
teachers_df = pd.DataFrame({
    'teacher_id': teacher_id,
    'teacher_name': teacher_names_with_titles,
    'teacher_email' : email,
    'salary_range': salary_range,
    'department': department,
    'qualification': qualification,
    'employmentStatus': employment_status,
    'gender': gender,
    'yearsOfExperience': years_of_experience,
    'phone_number': phone_number
})


# Dictionary of subjects categorized into junior and senior classes
subjects_dict = {
    'junior': [
        'Mathematics', 'English Language', 'Basic Science', 'Basic Technology', 
        'Civic Education', 'Agricultural Science', 'Social Studies', 
        'Business Studies', 'Home Economics', 'Physical and Health Education'
    ],
    'senior': [
        'Mathematics', 'English Language', 'Biology', 'Chemistry', 'Physics', 
        'Further Mathematics', 'Economics', 'Geography', 'Civic Education', 
        'Agricultural Science', 'History', 'Government', 'Literature in English', 
        'Computer Studies', 'Commerce', 'Financial Accounting', 'French', 
        'Christian Religious Studies', 'Islamic Religious Studies'
    ]
}

subject_ids = [f'subject_{i}' for i in range(1, len(subjects_dict['junior'] + subjects_dict['senior']) + 1)]

# Create DataFrame for subjects with corresponding IDs
subject_df = pd.DataFrame({
    'subject_id': subject_ids,
    'subject': subjects_dict['junior'] + subjects_dict['senior']
})

student_id_attendance = [i for i in range(no_of_students * no_of_attendance_records)]
student_id_link = np.repeat(student_id, no_of_attendance_records)

attendance_date = np.array([datetime(2024, random.randint(1, 12), random.randint(1, 28)) 
                            for _ in range(no_of_students * no_of_attendance_records)])

attendance_status = np.random.choice(['Present', 'Absent'], size=no_of_students * no_of_attendance_records, p=[0.85, 0.15])

attendance_data = {
    'attendance_id': student_id_attendance,
    'student_id': student_id_link,
    'date': attendance_date,
    'status': attendance_status
}

attendance_df = pd.DataFrame(attendance_data)
attendance_df['status'] = attendance_df['status'].map({'Present': 1, 'Absent': 0})
total_possible_days = 70

attendance_summary = attendance_df.groupby('student_id')['status'].sum().reset_index(name='totalDaysPresent')
attendance_summary['attendancePercentage_overall'] = (attendance_summary['totalDaysPresent'] / total_possible_days) * 100
attendance_summary['attendancePercentage_overall'] = np.clip(attendance_summary['attendancePercentage_overall'], 0, 100)

# Academic data for each student based on subjects
data = {
    'student_id': [],
    'subject_id': [],
    'attendance_percentage': [],
    'assignment_score': [],
    'exam_score': [],
    'final_grade': []
}

for student in student_id:
    class_level = random.choice(['junior', 'senior'])
    
    # Each student gets a random number of subjects (7-10)
    num_subjects = random.randint(8, 10)
    selected_subjects = random.sample(subjects_dict[class_level], k=num_subjects)
    
    for subject in selected_subjects:
        # Increase range for more variability
        attendance_percentage = round(random.uniform(50, 100), 2)  # Allow lower attendance
        assignment_score = round(random.uniform(40, 100), 2)  # Wider range for assignments
        exam_score = round(random.uniform(40, 100), 2)  # Exams can vary even more
        
        # Adding a small amount of random noise for more variability
        noise = random.uniform(-3, 3) 
        
        # Adjusting the formula to change the contribution of each score
        final_grade = round((attendance_percentage * 0.2) + (assignment_score * 0.2) + (exam_score * 0.6) + noise, 2)
        
        # Ensuring the final grade stays within 0-100
        final_grade = max(0, min(100, final_grade))

        data['student_id'].append(student)
        data['subject_id'].append(subject_df.query("subject == @subject")['subject_id'].values[0])
        data['attendance_percentage'].append(attendance_percentage)
        data['assignment_score'].append(assignment_score)
        data['exam_score'].append(exam_score)
        data['final_grade'].append(final_grade)

academic_df = pd.DataFrame(data)
academic_df['academic_record_id'] = academic_df.index

aggregated_academic_df = academic_df.groupby('student_id')['final_grade'].mean().reset_index()

aggregated_academic_df.columns = ['student_id', 'average_Final_Grade']

#teacher evaluation
teacher_evaluation_data = {
    'student_id': [],
    'subject_id': [],
    'evaluation_date': [],
    'comments': [],
    'point': []
}

comments_mapping = {
    'Excellent performance': 10,
    'Good performance': 8,
    'Fair performance': 6,
    'Average performance': 4,
    'Below average performance': 2,
    'Very bad performance': 0
}

for student in student_id:
    class_level = random.choice(['junior', 'senior'])
    selected_subjects = random.sample(subjects_dict[class_level], k=random.randint(8, 10))
    
    for subject in selected_subjects:
        evaluation_date = datetime.now() - timedelta(days=random.randint(0, 30))
        comment = random.choices(list(comments_mapping.keys()), weights = [0.3,0.15, 0.15,0.05,0.15,0.2])
        
        teacher_evaluation_data['student_id'].append(student)
        # teacher_evaluation_data['subject'].append(subject)
        teacher_evaluation_data['subject_id'].append(subject_df.query("subject == @subject")['subject_id'].values[0])
        teacher_evaluation_data['evaluation_date'].append(evaluation_date.strftime('%Y-%m-%d'))
        teacher_evaluation_data['comments'].append(comment[0])
        teacher_evaluation_data['point'].append(comments_mapping[comment[0]])

teacher_evaluation_df = pd.DataFrame(teacher_evaluation_data)

teacher_evaluation_df['point'] = teacher_evaluation_df['comments'].map(comments_mapping)
aggregated_teacher_evaluation = teacher_evaluation_df.groupby('student_id')['point'].mean().reset_index()
aggregated_teacher_evaluation.columns = ['student_id', 'average_Teacher_Evaluation']

# Class-Subject-Teacher Assignment
data = {
    'class': [],
    'subject_id': [],
    'teacher_id': []
}

classes = ['Jss 1', 'Jss 2', 'Jss 3', 'SS1', 'SS2', 'SS3']

for class_name in classes:
    class_type = 'junior' if 'Jss' in class_name else 'senior'
    num_subjects = random.randint(5, 8)
    selected_subjects = random.sample(subject_ids, k=num_subjects)
    selected_teachers = random.choices(teacher_id, k=num_subjects)
    
    for subj, teacher in zip(selected_subjects, selected_teachers):
        data['class'].append(class_name)
        data['subject_id'].append(subj)
        data['teacher_id'].append(teacher)

class_assignment_df = pd.DataFrame(data)


merged_df = pd.merge(students_df, parent_df, on='student_id', how='left',  suffixes=('', '_parent'))
merged_df = pd.merge(merged_df, academic_df, on='student_id', how='left',  suffixes=('', '_academic'))
merged_df = pd.merge(merged_df, teacher_evaluation_df, on=['student_id','subject_id'], how='inner',  suffixes=('', '_teacher_evaluation'))
merged_df = pd.merge(merged_df, subject_df, on=['subject_id'], how='left',  suffixes=('', '_subject'))
# merged_df = pd.merge(merged_df, class_assignment_df, on=['class','subject_id'], how='left',  suffixes=('', '_class'))
# merged_df = pd.merge(merged_df, teachers_df, on='teacher_id', how='inner',  suffixes=('', '_teacher'))

final_df = pd.merge(merged_df, attendance_summary, on='student_id', how='left',  suffixes=('', '_attendance_summary'))


# Define thresholds
attendance_threshold = 50
exam_score_threshold = 50
assignment_score_threshold = 50
final_grade_threshold = 50

# You can add more complex conditions based on study time, anxiety, etc.final
final_df['target'] = np.where(
    (final_df['attendance_percentage'] >= attendance_threshold) & (final_df['point'] > 5) & (final_df['internetAccess'] == True) & (final_df['access_to_constant_electricity'] == True) & (final_df['anxiety_before_during_exams'] == False) &
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


#storing datasets
students_data = os.path.join(current_dir,'datasets','students_data.csv')
students_df.to_csv(students_data, index=False)

parents_data = os.path.join(current_dir,'datasets','parents_df.csv')
parent_df.to_csv(parents_data, index=False)

teachers_data = os.path.join(current_dir,'datasets','teachers_df.csv')
teachers_df.to_csv(teachers_data, index=False)

att_data = os.path.join(current_dir,'datasets','attendance_df.csv')
attendance_df.to_csv(att_data, index=False)

att_data = os.path.join(current_dir,'datasets','attendance_summary.csv')
attendance_summary.to_csv(att_data, index=False)

att_data = os.path.join(current_dir,'datasets','student_academic_data.csv')
academic_df.to_csv(att_data, index=False)

att_data = os.path.join(current_dir,'datasets','aggregated_academic_df.csv')
aggregated_academic_df.to_csv(att_data, index=False)

att_data = os.path.join(current_dir,'datasets','teacher_evaluation_data.csv')
teacher_evaluation_df.to_csv(att_data, index=False)

att_data = os.path.join(current_dir,'datasets','aggregated_teacher_evaluation.csv')
aggregated_teacher_evaluation.to_csv(att_data, index=False)

att_data = os.path.join(current_dir,'datasets','class_to_teacher.csv')
class_assignment_df.to_csv(att_data, index=False)

att_data = os.path.join(current_dir,'datasets','subjects_offered.csv')
subject_df.to_csv(att_data, index=False)

final_df = final_df.sample(frac = 1, random_state = 42)
att_data = os.path.join(current_dir,'datasets','school_dataset.csv')
final_df.to_csv(att_data, index=False)

print('successfull')