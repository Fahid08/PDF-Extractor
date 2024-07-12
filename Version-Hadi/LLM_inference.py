from dotenv import load_dotenv
from groq import Groq
from openai.types.chat.completion_create_params import ResponseFormat
import os

load_dotenv('.env')

extracted_text_file_path = 'extracted_text.txt'

# Read the entire file into a variable
with open(extracted_text_file_path, 'r', encoding='utf-8') as file:
    text_data = file.read()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

delimiter = "####"
retrieval_message = """You will be given unstructured extracted text from a pdf (that initally contained both text data and structured tables).
The text you will be provided will include ENTITIES, for example,the amount of CO2 emitted from chimney 2 at night, AND THEIR CORRESPONDING 
MEASURED VALUES, your objective is to extract those entities and their values from the text provided and write them to separate json dictionaries
in the format {prompt:' ', completion: ''}. This output will be used to fine tune an OpenAI model. The prompt will resemble a query like - 
How much N2O was produced by this factory in 2022 and the completion will provide a suitable answer to this prompt, completing it.
Try to use the sentence the entity was found in to find the perfect entity name that represents it. The values  used in the completion should be 
either numerical amounts or something from which numerical figures can be inferred, such as a percentage, a portion or a fraction. Return only 
the dictionaries as output, each in a separate line. The output you return should start with the beginning of the first dictionary. The relevant text is given inside the delimitter ####.
"""
# Your output should be one line and not contain any escape sequences or new line characters.Moreover, 

model_name ="llama3-70b-8192" 
# response_format= ResponseFormat({ "type": "json_object" })

#model_name="mixtral-8x7b-32768"
chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"{retrieval_message}{delimiter}{text_data}{delimiter}",
                    }
                    ],
                model=model_name,
                # response_format=response_format,
)

# print(chat_completion.choices[0].message.content)
prompt_completions = chat_completion.choices[0].message.content
  
