from anthropic import Anthropic
from dotenv import load_dotenv
import os
import argparse
import base64

from prompts import SYSTEM_MESSAGE, USER_PROMPT


def encode_file(file_path):
    """
    Encodes any file to base64 and determines the correct media_type.

    Args:
        file_path (str): Path to the file

    Returns:
        tuple: (base64_string, media_type)
    """
    # Map file extensions to media types
    media_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.doc': 'application/msword',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    }

    # Get file extension and convert to lowercase
    _, ext = os.path.splitext(file_path.lower())

    # Determine media type based on file extension
    media_type = media_types.get(ext, 'application/octet-stream')

    # Read and encode the file
    with open(file_path, "rb") as f:
        encoded_file = base64.b64encode(f.read()).decode('utf-8')

    return encoded_file, media_type


def main():
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

    # Encode image with proper media type detection
    base64_image, image_media_type = encode_file(args["image"])

    # Encode guidelines if provided
    base64_guidelines = None
    guidelines_media_type = None
    if args["guidelines"]:
        base64_guidelines, guidelines_media_type = encode_file(args["guidelines"])

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
                "media_type": guidelines_media_type,
                "data": base64_guidelines
            }
        })

    # Add image content with correct media type
    messages[0]["content"].append({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": image_media_type,
            "data": base64_image
        }
    })

    try:
        # Make the API call
        response = client.messages.create(
            model=args["model"],
            system=SYSTEM_MESSAGE,
            messages=messages,
            max_tokens=4096
        )

        # Print the response
        print(response.content[0].text)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()