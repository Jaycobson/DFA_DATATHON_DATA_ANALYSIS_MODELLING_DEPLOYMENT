# import streamlit as st
# import pandas as pd
# import json
# from langchain_community.llms import OpenAI
# from langchain_experimental.agents import create_pandas_dataframe_agent
# # import environ


# API_KEY = "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"

# def decode_response(response: str) -> dict:
#     """This function converts the string response from the model to a dictionary object."""
#     return json.loads(response)

# def write_response(response_dict: dict):
#     """Write a response from an agent to a Streamlit app."""
#     if "answer" in response_dict:
#         st.write(response_dict["answer"])
#     if "bar" in response_dict:
#         data = response_dict["bar"]
#         df = pd.DataFrame(data["data"], columns=data["columns"])
#         st.bar_chart(df.set_index(df.columns[0]))
#     if "line" in response_dict:
#         data = response_dict["line"]
#         df = pd.DataFrame(data["data"], columns=data["columns"])
#         st.line_chart(df.set_index(df.columns[0]))
#     if "table" in response_dict:
#         data = response_dict["table"]
#         df = pd.DataFrame(data["data"], columns=data["columns"])
#         st.table(df)

# def create_agent():
#     """Create an agent from the CSV file."""
#     llm = OpenAI(openai_api_key=API_KEY)
#     return create_pandas_dataframe_agent(llm, data, verbose=False,allow_dangerous_code=True)

# def query_agent(agent, query):
#     """Query the agent and return the response."""
#     prompt = f"""
#         For the following query, if it requires drawing a table, reply as follows:
#         {{"table": {{"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}}}

#         If the query requires creating a bar chart, reply as follows:
#         {{"bar": {{"columns": ["A", "B", "C", ...], "data": [[label1, value1], [label2, value2], ...]}}}}
        
#         If the query requires creating a line chart, reply as follows:
#         {{"line": {{"columns": ["A", "B", "C", ...], "data": [[label1, value1], [label2, value2], ...]}}}}
        
#         There can only be two types of chart, "bar" and "line".
        
#         If it is just asking a question that requires neither, reply as follows:
#         {{"answer": "answer"}}
#         Example:
#         {{"answer": "The title with the highest rating is 'Gilead'"}}
        
#         If you do not know the answer, reply as follows:
#         {{"answer": "I do not know."}}
        
#         Return all output as a string.
        
#         All strings in "columns" list and data list, should be in double quotes,
        
#         For example: {{"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}}
        
#         Ensure that for bar and line charts, the data is a list of lists, where each inner list represents a data point with a label and a value.
        
#         Lets think step by step.
        
#         Below is the query.
#         Query: {query}
#     """
#     response = agent.run(prompt)
#     return response.__str__()

# # Streamlit app
# st.title("👨‍💻 Chat with your CSV")
# st.write("Please upload your CSV file below.")

# # File uploader
# data = pd.read_csv(r"C:\Users\DELL\Downloads\PROJECT_FOLDER\datasets\students_data.csv")

# # Query input
# query = st.text_area("Insert your query")

# if st.button("Submit Query", type="primary"):
#     if data is not None:
#         agent = create_agent()
#         response = query_agent(agent=agent, query=query)
#         try:
#             decoded_response = decode_response(response)
#             write_response(decoded_response)
#         except json.JSONDecodeError:
#             st.write(response)  # Fallback to display raw response if JSON decoding fails
#     else:
#         st.write("Please upload a CSV file before submitting a query.")
        
    


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

def create_chart(data, chart_type, columns):
    if chart_type == 'bar':
        fig = px.bar(data, x=data.index, y=columns[0])
    elif chart_type == 'line':
        fig = px.line(data, x=data.index, y=columns[0])
    elif chart_type == 'histogram':
        fig = px.histogram(data, x=columns[0])
    elif chart_type == 'scatter':
        if len(columns) == 2:
            fig = px.scatter(data, x=columns[0], y=columns[1])
        else:
            st.write("Scatter plot requires exactly two columns.")
            return
    elif chart_type == 'box':
        fig = px.box(data, y=columns[0])
    elif chart_type == 'pie':
        fig = px.pie(data, values=columns[0], names=data.index)
    elif chart_type == 'heatmap':
        if len(columns) >= 2:
            fig = px.imshow(data[columns].corr())
        else:
            st.write("Heatmap requires at least two columns.")
            return
    else:
        st.write(f"Chart type '{chart_type}' not recognized.")
        return

    st.plotly_chart(fig)

def get_stats(data, column):
    stats = data[column].describe()
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Statistic', 'Value']),
        cells=dict(values=[stats.index, stats.values])
    )])
    st.plotly_chart(fig)

def get_info(data):
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

def get_correlation(data):
    corr = data.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig)

def process_query(data, query):
    query = query.lower()
    
    chart_types = {
        'bar': 'bar', 'line': 'line', 'table': 'table',
        'histogram': 'histogram', 'scatter': 'scatter',
        'box': 'box', 'boxplot': 'box', 'pie': 'pie',
        'heatmap': 'heatmap'
    }
    chart_type = next((chart_types[word] for word in query.split() if word in chart_types), None)
    
    if 'stats' in query or 'statistics' in query:
        columns = [col for col in data.columns if col.lower() in query]
        if columns:
            get_stats(data, columns[0])
        else:
            st.write("Please specify a column for statistics.")
    elif 'info' in query or 'information' in query:
        get_info(data)
    elif 'correlation' in query or 'correlate' in query:
        get_correlation(data)
    elif chart_type:
        columns = [col for col in data.columns if col.lower() in query]
        if columns or chart_type == 'table':
            if chart_type == 'table':
                st.write(data)
            else:
                create_chart(data, chart_type, columns)
        else:
            st.write(f"Please specify column(s) for the {chart_type} chart.")
    else:
        st.write("I couldn't understand the query. Please try again with more specific terms.")

# Streamlit app
st.title("👨‍💻 Advanced CSV Analyzer with Plotly")
st.write("Please upload your CSV file below.")



data = pd.read_csv(r'C:\Users\DELL\Downloads\PROJECT_FOLDER\datasets\school_dataset.csv')
st.write("Data loaded successfully!")

# Display available columns
st.write("Available columns:")
st.write('some columns are',", ".join(data.columns))

# Query input
query = st.text_area("Insert your query")

if st.button("Submit Query", type="primary"):
    process_query(data, query)
