from anthropic import Anthropic
from dotenv import load_dotenv
import os
import argparse

from prompts import SYSTEM_MESSAGE, USER_PROMPT
from utils.base64_encodings import encode_image, encode_file

parser = argparse.ArgumentParser(prog="Claude API image client")

parser.add_argument("-i", "--image", required=True, help="path to input image file")
parser.add_argument("-g", "--guidelines", required=False, help="path to input guidelines pdf file")
parser.add_argument("-m", "--model", required=True, type=str,
                    choices=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
                             "claude-3-5-sonnet-20240620", "claude-3-7-sonnet-20250219"],
                    help="Claude model to use")
args = vars(parser.parse_args())

# loading .env file
load_dotenv()

base64_image = encode_image(args["image"])
base64_guidelines = encode_file(args["guidelines"]) if args["guidelines"] else None

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

# Prepare messages
messages = [
    {
        "role": "user",
        "content": []
    }
]

# Add text content
messages[0]["content"].append({
    "type": "text",
    "text": USER_PROMPT
})

# Add guidelines PDF if provided
if base64_guidelines:
    messages[0]["content"].append({
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64_guidelines
        }
    })

# Add image content
messages[0]["content"].append({
    "type": "image",
    "source": {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": base64_image
    }
})

# Make the API call
response = client.messages.create(
    model=args["model"],
    system=SYSTEM_MESSAGE,
    messages=messages,
    max_tokens=4096
)

# Print the response
print(response.content[0].text)