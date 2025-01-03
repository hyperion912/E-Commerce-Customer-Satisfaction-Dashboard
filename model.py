from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')


df = pd.read_csv("Customer_support_data.csv")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.5, max_tokens=None, timeout=None)

agent = create_pandas_dataframe_agent(llm, df,verbose=True, allow_dangerous_code=True)

agent.invoke("which product type has highest number of complaints")