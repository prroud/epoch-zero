import uuid
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = init_chat_model('google_genai:gemma-4-31b-it')


class IntentClassifier(BaseModel):
    message_intent: Literal['chat', 'knowledge', 'code'] = Field(
        ...,
        description='Classify whether the user wants to just chat, ask for knowledge' \
        'or change code in the project.'
    )


class State(TypedDict):
    messages = Annotated[list, add_messages]
    message_intent = str | None


def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier)
    result = structured_llm.invoke([
        {'role':'system',
         'content':'Determine whether user wants to chat, retrieve or code.'},
        {'role': 'user', 'content': state.messages[-1].content}
    ])

    return {'message_intent': result.message_intent}


def prompt_llm_chat(state: State):
    messages = [{'role': 'system', 'content':'You are talkative fun chatbot'}] + state['messages']
    response = llm.invoke(messages)

    return {'messages': {'role': 'assistant', 'content':response.content}}

def prompt_llm_rag(state: State):
    messages = [{'role': 'system', 'content':'No matter what user says, always say "I am the RAG agent"'}] + state['messages']
    response = llm.invoke(messages)

    return {'messages': {'role': 'assistant', 'content':response.content}}

def prompt_llm_code(state: State):
    messages = [{'role': 'system', 'content':'No matter what user says, always say "I am the CODING agent"'}] + state['messages']
    response = llm.invoke(messages)

    return {'messages': {'role': 'assistant', 'content':response.content}}
