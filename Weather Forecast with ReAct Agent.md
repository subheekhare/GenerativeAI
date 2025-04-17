import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import requests
from typing import List, Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# setting API KEYs
os.environ["OPENWEATHERMAP_API_KEY"] = OPENWEATHERMAP_API_KEY
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# defining the tools
@tool
def search_web(query: str) -> list:
    """Search the web for a query"""
    tavily_search = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=2, search_depth='advanced', max_tokens=1000)
    results = tavily_search.invoke(query)
    return results

# defining the tools
@tool
def open_weather(query: str) -> list:
  """Search weatherapi to get the current weather"""
  weather = OpenWeatherMapAPIWrapper()
  try:
    endpoint = f"http://api.openweathermap.org/data/2.5/weather?q={query}&APPID={OPENWEATHERMAP_API_KEY}"
    response = requests.get(endpoint)
    data = response.json() 
    return data
  except:
    return "Weather Data Not Found" 

# Print Stream
def print_stream(stream):
    # message =''
    for s in stream:
        message = s["messages"][-1]
    # if isinstance(message, tuple):
    #   print(message)
    # else:
    return message.content


# Llama Model            
llm = ChatOpenAI(base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY,
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
# combination of Tools
tools = [search_web, open_weather]

# Tools bind with LLM
llm_with_tools = llm.bind_tools(tools)

# system prompt is used to inform the tools available to when to use each
system_prompt = """Act as a helpful assistant.
    Use the tools at your disposal to perform tasks as needed.
        - open_weather: whenever user asks get the weather of a place.
        - search_web: whenever user asks for information on current events or if you don't know the answer.
    Use the tools only if you don't know the answer.
    """
# Creating react agent
weather_agent = create_react_agent(model=llm, tools=tools, state_modifier=system_prompt)

# inputs = {"messages": [("user", "What is the current weather in Gurgaon today")]}
# print_stream(weather_agent.stream(inputs, stream_mode="values"))

def run_weather_app():
  st.title("Weather Forecast")
  city=st.text_input("Enter the City")
  st.button("Get Weather")
  query={"messages": [("user", f"What's is the weather of {city} in next 3hrs.")]}
  report=print_stream(weather_agent.stream(query, stream_mode="values"))
  st.text(report)
  
if __name__ == '__main__':
  run_weather_app()

$ streamlit run app.py

<img width="997" alt="image" src="https://github.com/user-attachments/assets/4c7ca5c7-1e35-49e1-8640-fb849d8c7943" />
