from dotenv import load_dotenv 

from langchain.agents import create_agent
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')
"""
texts = [
    'Apple makes very good computers.',
    'I believe Apple is innovative!',
    'I love apples',
    'I am a fan of MacBooks.',
    'I enjoy oranges',
    'I like Lenovo Thinkpads.',
    'I think pears taste very good.'
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)

#print(vector_store.similarity_search('Apples are my favourite food.', k=7))
print(vector_store.similarity_search('Linux is a great operating system.', k=7))
"""

texts = [
    'I love apples.',
    'I enjoy oranges.',
    'I think pears taste very good.',
    'I hate bananas.',
    'I dislike raspberries.',
    'I despise mangos.',
    'I love Linux.',
    'I hate Windows.'
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)

print(vector_store.similarity_search('What fruits does the person like?', k=3))
print(vector_store.similarity_search('What fruits does the person hate?', k=3))

retriever = vector_store.as_retriever(search_kwargs={'k': 3})

retriever_tool = create_retriever_tool(retriever, name='kb_search', description='Search the small product / fruit knowledge base for information')

agent = create_agent(
    model='google_genai:gemma-4-31b-it',
    tools = [retriever_tool],
    system_prompt=('You are helpful assistant. For questions about Macs, apples or laptops. First call the kb_search tool' \
    'to retrieve context, then answer succintly. Maybe you have to see it multiple times before answering')
)

result = agent.invoke({
    'messages': [{'role': 'user', 'content':'What three fruits does the person like and what three fruits does the person dislike?'}]
})

print(result)
print(result['messages'][-1].content)


