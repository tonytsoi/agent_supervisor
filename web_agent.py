from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
import os
import credentials

os.environ['AWS_DEFAULT_REGION'] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
tavily_api_key = credentials.tavily_api_key

llm = init_chat_model("amazon.nova-pro-v1:0",
                            model_provider="bedrock_converse")

web_search = TavilySearch(max_results=3, tavily_api_key=tavily_api_key)

web_agent = create_react_agent(
    model=llm,
    tools=[web_search],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="web_agent",
)