import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = init_chat_model(
    model = 'google_genai:gemma-4-31b-it',
    temperature = 0.1
)

"""
We can also invoke on conversation

conversation = [
    SystemMessage("You are a helpful assistant for questions regarding animals"),
    HumanMessage("What is Python?"),
    AIMessage("Python is a snake"),
    HumanMessage("Where do they live?")
]

response = model.invoke(conversation)
"""

"""
response = model.invoke('Hello, where the hedgehog is going at night?')
print(response)
print(response.content)
"""

for chunk in model.stream('Hello, where the hedgehog is going at night?'):
    print(chunk.text, end='', flush=True)