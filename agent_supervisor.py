from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from web_agent import web_agent
from sql_agent import sql_agent
import streamlit as st
import credentials
import os
import uuid

os.environ['AWS_DEFAULT_REGION'] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key

thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}
tavily_api_key = credentials.tavily_api_key

def generate_response(input_text):
    # Create the agent
    llm = init_chat_model("amazon.nova-pro-v1:0",
                            model_provider="bedrock_converse")

    supervisor = create_supervisor(
        model=llm,
        agents=[web_agent, sql_agent],
        prompt=(
            "You are a supervisor managing two agents:\n"
            "- a web search agent. Assign research-related tasks to this agent\n"
            "- a SQL agent. Assign SQL-related tasks regarding the bike store to this agent\n"
            "Assign work to one agent at a time, do not call agents in parallel.\n"
            "Do not do any work yourself."
        ),
        add_handoff_back_messages=True,
        output_mode="full_history"
    ).compile()

    # Use the agent
    config = {"configurable": {"thread_id": "abc123"}}
    for step in supervisor.stream(
            {"messages": [HumanMessage(content=f"{input_text}")]},
            config,
            stream_mode="values",
    ):
        if step["messages"][-1].type != 'human':
            try:
                for sentence in step["messages"][-1].content[0]['text'].split("/n"):
                    yield sentence + "  \n\n"
            except:
                for sentence in step["messages"][-1].content.split("/n"):
                    yield sentence + "  \n\n"


st.title("Multi-Agent Supervisor")
st.caption("Powered by Amazon Nova Pro")
st.image("multi_agent.jpg")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message['content'])

# Accept user input
if prompt := st.chat_input("What do you want to ask?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(generate_response(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})