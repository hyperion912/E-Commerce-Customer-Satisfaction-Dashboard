import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

st.set_page_config(page_title="E-Commerce Dashboard", page_icon="📊", layout="wide")

load_dotenv()
os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('Customer_support_data.csv')  # Replace with your dataset path
    data['order_date_time'] = pd.to_datetime(data['order_date_time'], errors='coerce')
    data['Survey_response_Date'] = pd.to_datetime(data['Survey_response_Date'], errors='coerce')
    data['response_time'] = (data['Survey_response_Date'] - data['order_date_time']).dt.days
    data['CSAT_category'] = pd.cut(data['CSAT Score'], bins=[0, 2, 4, 5], labels=['Low', 'Medium', 'High'])
    data['Customer_City'].fillna('Unknown', inplace=True)
    data['Product_category'].fillna('Unknown', inplace=True)
    return data

data = load_data()

# Initialize LangChain Agent
@st.cache_resource
def initialize_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.5, max_tokens=None, timeout=None)
    agent = create_pandas_dataframe_agent(llm, data, verbose=True, allow_dangerous_code=True)
    return agent

agent = initialize_agent()

def reset_agent():
    st.cache_data.clear()
    st.cache_resource.clear() 
    return initialize_agent()

st.title("E-Commerce Customer Satisfaction Dashboard")

# Sidebar
st.sidebar.header("Filter Options")
channel_filter = st.sidebar.multiselect(
    "Select Channels:",
    options=data['channel_name'].unique(),
    default=data['channel_name'].unique(),
    help="Filter by the communication channels used by customers"
)
category_filter = st.sidebar.multiselect(
    "Select Product Categories:",
    options=data['Product_category'].unique(),
    default=data['Product_category'].unique(),
    help="Filter by product categories"
)
csat_filter = st.sidebar.multiselect(
    "Select CSAT Categories:",
    options=data['CSAT_category'].dropna().unique(),
    default=data['CSAT_category'].dropna().unique(),
    help="Filter by Customer Satisfaction (CSAT) levels"
)

# Filter data
filtered_data = data[
    (data['channel_name'].isin(channel_filter)) &
    (data['Product_category'].isin(category_filter)) &
    (data['CSAT_category'].isin(csat_filter))
]

# Chatbot Section
st.title("💬 Chat with the AI Agent")

st.subheader("Ask Questions about the Data (powered by Gemini)")
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Processing your query..."):
        agent = reset_agent()
        response = agent.invoke(query)
    st.subheader("Response")
    st.write(response)

# Display filtered data
st.subheader("Filtered Data Preview")
st.write(f"Showing {filtered_data.shape[0]} records")
st.dataframe(filtered_data)

# Insights: CSAT Distribution
st.subheader("CSAT Score Distribution")
csat_counts = filtered_data['CSAT_category'].value_counts()
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=csat_counts.index, y=csat_counts.values, ax=ax, palette="coolwarm")
ax.set_title("CSAT Score Distribution", fontsize=16)
ax.set_xlabel("CSAT Category", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
st.pyplot(fig)

# Insights: Response Time Analysis
st.subheader("Response Time vs CSAT Category")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='CSAT_category', y='response_time', data=filtered_data, ax=ax, palette="coolwarm")
ax.set_title("Response Time vs CSAT Category", fontsize=16)
ax.set_xlabel("CSAT Category", fontsize=12)
ax.set_ylabel("Response Time (days)", fontsize=12)
st.pyplot(fig)

# Channel-wise Satisfaction
st.subheader("Average CSAT Score by Channel")
channel_csat = filtered_data.groupby('channel_name')['CSAT Score'].mean().sort_values()
st.bar_chart(channel_csat)

# Product Category Issues
st.subheader("Issues by Product Category and CSAT Category")
category_csat = filtered_data.groupby(['Product_category', 'CSAT_category']).size().unstack()
st.bar_chart(category_csat)

# Agent Performance
st.subheader("Agent-wise CSAT Performance")
agent_csat = filtered_data.groupby('Agent_name')['CSAT Score'].mean().sort_values(ascending=False)
st.bar_chart(agent_csat.head(10))

# Summary Statistics
st.subheader("Summary Statistics")
st.write(filtered_data.describe())

st.title("📊 Prompt-Based Graphs")

# Prompt input for generating graphs
st.subheader("Generate Graphs with a Custom Prompt")
user_prompt = st.text_area("Enter your prompt (e.g., 'Generate a heatmap of average CSAT score by product category'):")

user_prompt = "write the code for" + user_prompt

if st.button("Generate Graph"):
    if user_prompt:
        try:
            # Use the agent to execute the prompt
            agent = reset_agent()
            result = agent.invoke(user_prompt)
            # print(result)
            exec(str(result))
            st.pyplot(plt)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a prompt to generate a graph.")
