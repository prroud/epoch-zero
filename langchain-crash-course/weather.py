from dataclasses import dataclass 

import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime 
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

@dataclass
class Context:
    user_id: str

@dataclass 
class ResponseFormat:
    summary: str 
    temperature_celsius: float 
    temperature_fahrenheit: float 
    humidity: float

"""
@tool('get_weather', description='Return weather information for a given city', return_direct=False)
def get_weather(city: str) -> dict:
    response = requests.get(f'https://wttr.in/{city}?format=j1', timeout=10)
    response.raise_for_status()
    data = response.json()
    
    current = data['current_condition'][0]
    
    return {
        'city': city,
        'temp_C': current['temp_C'],
        'feels_like_C': current['FeelsLikeC'],
        'condition': current['weatherDesc'][0]['value'],
        'humidity_percent': current['humidity'],
        'wind_kmph': current['windspeedKmph']
    }
"""

@tool('get_weather', description='Return weather information for a given city', return_direct=False)
def get_weather(city: str) -> dict:
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()


@tool('locate_user', description="Look up a user's city based on the context")
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case 'ABC123':
            return 'Vienna'
        case 'XYZ456':
            return 'London'
        case 'HJKL111':
            return 'Paris'
        case _:
            return 'Unknown'


agent = create_agent(
    model = 'google_genai:gemma-4-31b-it',
    tools = [get_weather],
    system_prompt='You are a helpful weather assistant, who always cracks jokes and is humorous while remaining helpful'
)

response = agent.invoke({
    'messages':[
        {'role': 'user', 'content': 'What is the weather like in Krakow?'}
    ]
})

print(response)
print(response['messages'][-1].content)