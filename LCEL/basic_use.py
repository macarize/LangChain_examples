import os
import openai
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv, find_dotenv
import time

openai.api_key = os.environ.get('OPENAI_API_KEY')

'''
Simple Chain
'''
model = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", max_tokens=150)

prompt = ChatPromptTemplate.from_template(
    "Tell me a short advice about {topic}"
)

output_parser = StrOutputParser()
chain = prompt | model | output_parser

print(chain.invoke({"topic": "learning python"}))

'''
More complex chain
'''
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch

vectorstore = DocArrayInMemorySearch.from_texts(
    ['Practicing is the best way to learn.', 'Bristol is famous for its thriving underground music scene.'],
    embedding=OpenAIEmbeddings(),
)

retriever = vectorstore.as_retriever()

print(retriever.get_relevant_documents("Where is the best place to start my DJ career?"))

template = """"
Answer the question based only on the following context: {context}
question: {question}
"""

promt = ChatPromptTemplate.from_template(template)

from langchain.schema.runnable import RunnableMap
chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"],
}) | promt | model | output_parser

print(chain.invoke({"question": "Where is the best place to start my DJ career?"}))

"""
Bind
"""
functions = [
    {
      "name": "weather_search",
      "description": "Search for weather given an airport code",
      "parameters": {
        "type": "object",
        "properties": {
          "airport_code": {
            "type": "string",
            "description": "The airport code to get the weather for"
          },
        },
        "required": ["airport_code"]
      }
    }
  ]

promt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}")
    ]
)
model = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile").bind(functions=functions)
runnable = promt | model

print(runnable.invoke({"input": "What is the weather in San Francisco?"}))