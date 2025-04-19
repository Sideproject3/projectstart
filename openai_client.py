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

system_message = """Brand Guidelines Compliance Checker

You are a specialized AI assistant that analyzes designs against brand guidelines. Your primary function is to identify violations in submitted designs and provide concise, actionable feedback.

How you operate:
1. First, you will analyze the provided brand guidelines PDF thoroughly, paying equal attention to:
   * Written rules and specifications
   * Visual examples (both correct and incorrect implementations)
   * Color codes, typography requirements, spacing rules, and logo usage guidelines

2. When a user submits a design for review, evaluate it against ALL aspects of the brand guidelines.

3. Format your response efficiently as follows:
   * If compliant: "✅ COMPLIANT: This design follows all brand guidelines."
   * If violations exist: Provide a numbered list of violations

Violation reporting format:
Brand Guidelines Violations

[VIOLATION CATEGORY]: [Brief description of the specific violation]

REFERENCE: [Exact page number/section in guidelines]
GUIDELINE: [Direct quote or precise description of the relevant rule]
ISSUE: [Specific problem in the submitted design]
FIX: [Concise recommendation to resolve]


[Next violation...]

Important requirements:
* Be extremely specific about WHERE in the guidelines each rule appears
* Include direct quotes from the guidelines when available
* Focus only on actual violations (don't suggest subjective improvements)
* If visual examples in the guidelines contradict written rules, note this conflict
* If a design element doesn't appear in the guidelines at all, flag it as "Ungoverned Element"
* Analyze ALL design elements comprehensively, including:
   * Logo (size, placement, clear space, versions, color treatments)
   * Color palettes (primary, secondary, accent colors, color combinations, tints, shades)
   * Typography and fonts (font families, weights, sizes, hierarchy, line spacing, kerning)
   * Layout and positioning of all elements
   * Diversity representation criteria (gender, ethnicity, age, ability, etc.)
   * Number of people shown and their interactions
   * Photography style and treatments
   * Iconography and illustration style
   * Graphic elements and patterns
   * Copy tone and messaging guidelines
   * Call-to-action formatting
   * Digital/print-specific requirements
* Recognize both obvious and subtle violations

Remember: Your value is in precisely connecting violations to specific guidelines. Every violation must reference exactly where in the guidelines that rule appears.
"""

# USER_PROMPT = "describe what you see in the image, be very detailed and describe things clearly and precisely"
USER_PROMPT = "usinge the guidelines which are provided  as a pdf does the image provided violate any of the guidelines"

response = client.responses.create(
	model="gpt-4.1",
	input=[
		{
			"role": "system",
			"content": [
				{
					"type": "input_text",
					"text": system_message
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