import cv2
from pyzbar.pyzbar import decode
from datetime import datetime
import pandas as pd
from barcode import generate
from barcode.writer import ImageWriter
import io
import os
import numpy as np
import streamlit as st
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, 'datasets')
school_dataset = os.path.join(dataset_dir,'school_dataset.csv')
attendance_dataset = os.path.join(dataset_dir,'attendance_data.csv')
barcode_folder = os.path.join(current_dir, 'barcodes')
os.makedirs(barcode_folder, exist_ok=True)


df = pd.read_csv(school_dataset)
attendance_df = pd.read_csv(attendance_dataset)


def generate_barcode(student_id):
    barcode_file_path = os.path.join(barcode_folder, student_id)

    print(f"Barcode file will be saved at: {barcode_file_path}")
    # Check if the barcode already exists
    if os.path.exists(barcode_file_path):
        st.warning(f"Barcode already exists for Student ID: {student_id}. Not generating again.")
        return False  # Indicates that the barcode already exists

    # Generate the barcode
    rv = io.BytesIO()
    generate('code128', student_id, writer=ImageWriter(), output=rv)
    
    with open(f"{barcode_file_path}.png", 'wb') as f:
        f.write(rv.getvalue())
    
    # st.success(f"Barcode generated for Student ID: {student_id}")
    return True  # Indicates that the barcode was generated

# Function to log attendance if the student is valid
def log_attendance(student_id):
    global attendance_df
    current_time = datetime.now()
    # Check if student ID is already logged today
    if not attendance_df[(attendance_df['student_id'] == student_id) & 
                         (attendance_df['date'].str.startswith(current_time.strftime("%Y-%m-%d")))].empty:
        return "Marked Already", (0, 100, 255)  # Red for already marked
    else:
        # Auto-generate attendance_id based on the number of rows
        next_id = len(attendance_df)  # Increment based on the number of records

        # Create new row with auto-incrementing attendance_id
        new_row = {
            'attendance_id': next_id,
            'student_id': student_id, 
            'date': current_time.strftime("%Y-%m-%d %H:%M:%S"), 
            'status': 1
        }
        

        # Append the new row to the DataFrame
        attendance_df = pd.concat([attendance_df, pd.DataFrame([new_row])], ignore_index=True)
        attendance_df.to_csv(attendance_dataset,index=False)
        attendance_df.loc[len(attendance_df)] = [next_id, student_id, current_time.strftime("%Y-%m-%d %H:%M:%S"), 1]
        return "Marked Successfully", (0, 255, 0)  # Green for success



# Function to scan barcode and update attendance
def scan_barcode():
    cap = cv2.VideoCapture(0)  # Open the webcam (device 0)
    
    message = ""
    show_overlay = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and decode barcodes in the frame
        barcodes = decode(frame)
        for barcode in barcodes:
            # Extract barcode data
            
            barcode_data = barcode.data.decode('utf-8')
            
            barcode_data = barcode_data.replace('_','/')
           
            # Check if the barcode matches any student in the database
            if barcode_data in df['student_id'].values:
                print(barcode_data)
                message = log_attendance(barcode_data)[0]  # Log attendance
            else:
                message = "Student ID not in DB"  # Invalid ID
     
            show_overlay = True
            message_color = (0, 255, 0)  # Default to green
            # Create a copy of the frame for overlay
            overlay = frame.copy()
            overlay_start_time = datetime.now().timestamp()
            # Display overlay message if within the last 2 seconds
            if show_overlay and datetime.now().timestamp() - overlay_start_time < 10:
                # Add semi-transparent overlay
                cv2.addWeighted(overlay, 0.4, frame, 0.4, 0, frame)
                
                # Display the message at the center of the frame
                cv2.putText(frame, message, (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, message_color, 2)

            else:
                show_overlay = False  # Hide overlay after 2 seconds

        # Display the webcam feed with the message
        # cv2.putText(frame, message, (10, 50), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Barcode Scanner', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
    return barcode_data


# # Generate barcodes for all students
# if st.button("Generate Barcodes for All Students"):
#     # Progress bar
#     progress_bar = st.progress(0)  # Initialize progress bar
#     total_students = len(df['student_id'].unique())

#     for idx, student_id in enumerate(df['student_id'].unique()):
#         id = student_id.replace('/', '_')  # Handle special characters in student ID
#         generate_barcode(id)  # Call the barcode generation function

#         # Update progress bar
#         progress = (idx + 1) / total_students  # Calculate progress
#         progress_bar.progress(progress)  # Update progress bar

#     st.success("All barcodes generated!")
#     st.balloons()  # Show success balloons

# # Button to start scanning
# if st.button("Scan barcode"):
#     st.text("Please scan the barcode...")
#     scan_barcode()  # Start scanning when button is pressed
