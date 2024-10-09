import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, 'datasets')
data = os.path.join(dataset_dir, 'school_dataset.csv')

def to_plot():
    # Load the data
    df = pd.read_csv(data)

    # Title for the Streamlit app
    st.subheader("Student Performance Analysis")
    st.write('------------------------------')
    # Color Palette
    bright_colors = px.colors.qualitative.Set2  # Bright color scheme

    # Doughnut Chart
    # Create the doughnut chart
    labels = df['class'].value_counts().index
    values = df['class'].value_counts().values

    doughnut_fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, 
                                            marker=dict(colors=bright_colors))])
    doughnut_fig.update_layout(
        # title_text="Distribution of Students In Each Class",
        width=500,
        height=500,
        annotations=[dict(text='Classes', x=0.5, y=0.5, font_size=17, showarrow=False)],
        legend=dict(
            x=1.2,
            y=0.5,
            traceorder="normal",
            font=dict(size=18),
        )
    )

    # Display Doughnut Chart
    # Use HTML within st.markdown to make the text smaller
    st.markdown("<h3>1. Distribution of Students In Each Class</h3>", unsafe_allow_html=True)

    st.plotly_chart(doughnut_fig, use_container_width=True)
# Group the data by 'class' and 'gender' to get the count of students


    grouped_df = df.groupby(['class', 'gender']).size().reset_index(name='count')

    # Sunburst Chart
    sunburst_fig = px.sunburst(
        grouped_df,
        width=400,
        height=400,
        path=['class', 'gender'],  # Class and gender hierarchy
        values='count',  # Use the count for the values
        color='gender',  # Color by gender
        # title="Sunburst Chart of Student Count by Class and Gender",
        hover_data={'count': True},  # Show count on hover
        color_discrete_sequence=bright_colors  # Use the same bright color scheme
    )

    sunburst_fig.update_layout(
        title={
            # 'text': "Sunburst Chart of Class and Gender with Student Count",
            # 'yanchor': 'top'
        },
        margin=dict(t=50, l=25, r=25, b=25),
        uniformtext=dict(minsize=10, mode='hide')  # Hide overlapping text for clarity
    )

    # Display Sunburst Chart
    st.markdown("<h3>2. Distribution of Students In Each Class and Gender</h3>", unsafe_allow_html=True)
    st.plotly_chart(sunburst_fig, use_container_width=True)


    # Gender Distribution Column Chart
    samp = df['gender'].value_counts().reset_index()
    gender_chart_fig = px.bar(
        samp,
        y='count',
        x='gender',
        width=400,
        height=400,
        labels={'gender': 'Gender', 'count': 'Count'},
        # title='Gender Distribution in the Dataset',
        color='gender',
        color_discrete_sequence=bright_colors  # Use the same bright color scheme
    )

    gender_chart_fig.update_layout(
        xaxis_title='Gender',
        yaxis_title='Count',
        # title_x=0.5,
        showlegend=True,
        legend=dict(
            x=1.2,
            y=0.5,
            traceorder="normal",
            font=dict(size=18),
        )
    )

    # Display Gender Distribution Chart
    st.markdown("<h3>3. Distribution of Students per Gender</h3>", unsafe_allow_html=True)
    st.plotly_chart(gender_chart_fig, use_container_width=True)

    # Gender Distribution Column Chart
    samp = df['target'].value_counts().reset_index()
    gender_chart_fig = px.bar(
        samp,
        y='count',
        x='target',
        width=400,
        height=400,
        labels={'target': 'Pass/Fail', 'count': 'Count'},
        # title='Success Distribution in the Dataset',
        color='target',
        color_discrete_sequence=bright_colors  # Use the same bright color scheme
    )

    gender_chart_fig.update_layout(
        xaxis_title='Pass/Fail',
        yaxis_title='Count',
        # title_x=0.5,
        showlegend=True,
        legend=dict(
            x=1.2,
            y=0.5,
            traceorder="normal",
            font=dict(size=18),
        )
    )

    # Display Gender Distribution Chart
    st.markdown("<h3>4. Success Distribution of Students</h3>", unsafe_allow_html=True)
    st.plotly_chart(gender_chart_fig, use_container_width=True)

    # Pass/Fail Funnel Chart
    grouped_df = df.groupby(['class', 'target']).size().reset_index(name='Count')
    total_students_per_class = grouped_df.groupby('class')['Count'].sum().reset_index(name='Total_Students')
    grouped_df = pd.merge(grouped_df, total_students_per_class, on='class')
    grouped_df = grouped_df.sort_values(by='Total_Students', ascending=False)

    funnel_fig = px.funnel(grouped_df,
                            x='Count',
                            y='class',
                           width=400,
                            height=400,
                            color='target',
                            # title="Funnel Chart of Pass and Fail Counts in Each Class",
                            labels={'Count': 'Number of Students', 'class': 'Class'},
                            color_discrete_map={'Pass': 'green', 'Fail': 'red'})  # Bright colors for pass/fail

    funnel_fig.update_layout(legend_title_text="Pass or Fail", 
                             legend=dict(
            x=1.2,
            y=0.5,
            traceorder="normal",
            font=dict(size=18),
                                 
        ))

    # Display Funnel Chart
    st.markdown("<h3>5. Success Distribution of Students per Class</h3>", unsafe_allow_html=True)

    st.plotly_chart(funnel_fig, use_container_width=True)


    # Scatter Chart for Final Grade vs. Exam Score
    scatter_fig = px.scatter(
        df,
        x='exam_score',  # Exam score on x-axis
        y='final_grade',  # Final grade on y-axis
        width=400,
        height=400,
        color='target',  # Color based on pass/fail
        # size='age',  # Size of points based on age
        # title="Scatter Chart of Final Grade vs. Exam Score",
        hover_name='class',  # Show class on hover
        color_discrete_sequence=bright_colors  # Use the same bright color scheme
    )

    scatter_fig.update_layout(
        title_x=0.5,
        xaxis_title='Exam Score',
        yaxis_title='Final Grade',
        legend=dict(
            x=1.2,
            y=0.5,
            traceorder="normal",
            font=dict(size=18),
        )
    )

    # Display Scatter Chart
    st.markdown("<h3>6. Relationship between Final Grade vs. Exam Score</h3>", unsafe_allow_html=True)
    st.plotly_chart(scatter_fig, use_container_width=True)



    
    # Gender Distribution Column Chart
    samp = df['ethnicity'].value_counts().reset_index()
    ethnicity_chart_fig = px.bar(
        samp,
        y='count',
        x='ethnicity',
        width=400,
        height=400,
        labels={'ethnicity': 'Ethnicity', 'count': 'Count'},
        # title='Gender Distribution in the Dataset',
        color='ethnicity',
        color_discrete_sequence=bright_colors  # Use the same bright color scheme
    )

    ethnicity_chart_fig.update_layout(
        xaxis_title='Ethnicity',
        yaxis_title='Count',
        # title_x=0.5,
        showlegend=True,
        legend=dict(
            x=1.2,
            y=0.5,
            traceorder="normal",
            font=dict(size=18),
        )
    )

    # Display Gender Distribution Chart
    st.markdown("<h3>7. Distribution of Students per Ethnicity</h3>", unsafe_allow_html=True)
    st.plotly_chart(ethnicity_chart_fig, use_container_width=True)


    
    # Gender Distribution Column Chart
    samp = df['disability_status'].value_counts().reset_index()
    disability_chart_fig = px.bar(
        samp,
        y='count',
        x='disability_status',
        width=500,
        height=400,
        labels={'disability_status': 'Disability Status', 'count': 'Count'},
        # title='Gender Distribution in the Dataset',
        color='disability_status',
        color_discrete_sequence=bright_colors  # Use the same bright color scheme
    )

    disability_chart_fig.update_layout(
        xaxis_title='Disability Status',
        yaxis_title='Count',
        # title_x=0.5,
        showlegend=True,
        legend=dict(
            x=1.2,
            y=0.5,
            traceorder="normal",
            font=dict(size=18),
        )
    )

    # Display Gender Distribution Chart
    st.markdown("<h3>8. Distribution of Students for Disability/Non-Disability</h3>", unsafe_allow_html=True)
    st.plotly_chart(disability_chart_fig, use_container_width=True)



        # Gender Distribution Column Chart
    samp = df['qualification'].value_counts().reset_index()
    qualification_chart_fig = px.bar(
        samp,
        y='count',
        x='qualification',
        width=500,
        height=400,
        labels={'qualification': 'Qualification', 'count': 'Count'},
        # title='Gender Distribution in the Dataset',
        color='qualification',
        color_discrete_sequence=bright_colors  # Use the same bright color scheme
    )

    qualification_chart_fig.update_layout(
        xaxis_title='Qualification',
        yaxis_title='Count',
        # title_x=0.5,
        showlegend=True,
        legend=dict(
            x=1.2,
            y=0.5,
            traceorder="normal",
            font=dict(size=18),
        )
    )

    # Display Gender Distribution Chart
    st.markdown("<h3>9. Qualifications of Parents</h3>", unsafe_allow_html=True)
    st.plotly_chart(qualification_chart_fig, use_container_width=True)


    


