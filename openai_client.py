from openai import OpenAI
from dotenv import load_dotenv
import os
import argparse
import base64

parser = argparse.ArgumentParser(prog="Open api image client")

parser.add_argument("-i", "--image", required=True, help="path to input image file")
parser.add_argument("-g", "--guidelines", required=False, help="path to input guidelines pdf file file")
args = vars(parser.parse_args())

# Function to encode the image
def encode_image(image_path):
	with open(image_path, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode("utf-8")
	
def encode_guidelines(file_path):
	with open(file_path, "rb") as f:
		data = f.read()
	return base64.b64encode(data).decode("utf-8")

# loading .env file 
load_dotenv()

base64_image = encode_image(args["image"])
base64_guidelines = encode_guidelines(args["guidelines"])

client = OpenAI(
 	api_key=os.environ.get("OPENAI_API_KEY"),
)

# USER_PROMPT = "describe what you see in the image, be very detailed and describe things clearly and precisely"
USER_PROMPT = "usinge the guidelines which are provided  as a pdf does the image provided violate any of the guidelines"

response = client.responses.create(
	model="gpt-4.1",
	input=[
		{
			"role": "user",
			"content": [
				{ "type": "input_text", "text": USER_PROMPT },
				{
					"type": "input_file",
					"filename": "abc.pdf",
					"file_data": f"data:application/pdf;base64,{base64_guidelines}",
				},
				{
					"type": "input_image",
					"image_url": f"data:image/jpeg;base64,{base64_image}",
					# there is low, high & auto
					"detail": "auto",
				},
			],
		}
	],
)


print(response.output_text)