import os
import openai
from typing import List
from pydantic import BaseModel, Field

openai.api_key = os.environ.get('OPENAI_API_KEY')

class pUser(BaseModel):
    name: str
    age: int    
    email: str

class Class(BaseModel):
    students: List[pUser]

obj = Class(
    students=[pUser(name="Jame", age=32, email="jane@gmail.com")]
)

from langchain.utils.openai_functions import convert_pydantic_to_openai_function

"""
Pydantic to OpenAI function definition
"""
class WeatherSearch(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str = Field(description="airport code to get weather for")

weather_function = convert_pydantic_to_openai_function(WeatherSearch)

from langchain_groq import ChatGroq

model = ChatGroq(model_name="llama-3.3-70b-versatile")
print(model.invoke("what is the weather in New York?", functions=[weather_function]))

model_with_function = model.bind(functions=[weather_function])
print(model_with_function.invoke("what is the weather in London?"))

"""
Forcing it to use a function
"""
model_with_forced_function = model.bind(functions=[weather_function], function_call={"name": "WeatherSearch"})
print(model_with_forced_function.invoke("Hi!"))

"""
Using it a chain
"""
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
prompt = ChatPromptTemplate.from_messages([
    ("system", "Your are a helpful assistant"),
    ("user", "{input}")
])
parser = StrOutputParser()

chain = prompt | model_with_function
print(chain.invoke({"input": "What is the weather in Korea?"}))