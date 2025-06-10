from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
import sqlite3
import os
import credentials

os.environ['AWS_DEFAULT_REGION'] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
tavily_api_key = credentials.tavily_api_key

llm = init_chat_model("amazon.nova-pro-v1:0",
                            model_provider="bedrock_converse")

connection = sqlite3.connect('./data/bike_store.db', check_same_thread=False)

engine = create_engine(
    "sqlite://",
    creator=lambda: connection,
    poolclass=StaticPool,
)

db = SQLDatabase(engine)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
memory = MemorySaver()

system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)

sql_agent = create_react_agent(llm,
                               tools,
                               prompt=system_prompt,
                               checkpointer=memory,
                               name="sql_agent")