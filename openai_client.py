from openai import OpenAI
from dotenv import load_dotenv
import os
import argparse

from prompts import SYSTEM_MESSAGE, USER_PROMPT
from utils.base64_encodings import encode_image, encode_file

parser = argparse.ArgumentParser(prog="Open api image client")

parser.add_argument("-i", "--image", required=True, help="path to input image file")
parser.add_argument("-g", "--guidelines", required=False, help="path to input guidelines pdf file file")
parser.add_argument("-m", "--model", required=True, type=str, choices=["gpt-4.1"], help="path to input guidelines pdf file file")
args = vars(parser.parse_args())

# loading .env file 
load_dotenv()

base64_image = encode_image(args["image"])
base64_guidelines = encode_file(args["guidelines"])

client = OpenAI(
 	api_key=os.environ.get("OPENAI_API_KEY"),
)

response = client.responses.create(
	model=args["model"],
	input=[
		{
			"role": "system",
			"content": [
				{
					"type": "input_text",
					"text": SYSTEM_MESSAGE
				}
			]
		},
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