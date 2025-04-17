from openai import OpenAI
from dotenv import load_dotenv
import os

# loading .env file 
load_dotenv()

client = OpenAI(
 	api_key=os.environ.get("OPENAI_API_KEY"),
)

response = client.responses.create(
	model="gpt-4.1",
	input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)