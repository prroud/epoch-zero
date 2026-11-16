from dotenv import load_dotenv 

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

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
    
]