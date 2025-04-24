from google import genai
from google.genai import types
import pathlib
import httpx

from dotenv import load_dotenv
import os
import argparse

from prompts import SYSTEM_MESSAGE, USER_PROMPT
from utils.base64_encodings import encode_image, encode_file

parser = argparse.ArgumentParser(prog="Open api image client")

parser.add_argument("-i", "--image", required=True, help="path to input image file")
parser.add_argument("-g", "--guidelines", required=False, help="path to input guidelines pdf file file")
parser.add_argument("-m", "--model", required=True, type=str, choices=["gem	ini-2.5-flash-preview-04-17", "gemini-2.0-flash", "gemini-2.5-pro-preview-03-25", "gemini-2.5-pro-exp-03-25", "gemini-1.5-flash"], help="path to input guidelines pdf file file")
args = vars(parser.parse_args())

# loading .env file 
load_dotenv()

with open(args["image"], 'rb') as f:
	img_bytes = f.read()

pdf_filepath = pathlib.Path(args["guidelines"])

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

response = client.models.generate_content(
	model=args["model"],
	config=types.GenerateContentConfig(
		system_instruction=SYSTEM_MESSAGE
	),
	contents=[
		types.Part.from_bytes(
			data=pdf_filepath.read_bytes(),
			mime_type='application/pdf',
		),
		types.Part.from_bytes(
			data=img_bytes,
			mime_type='image/jpeg',
		),
	USER_PROMPT]
)


print(response.text)