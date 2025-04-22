import base64


def encode_image(image_path): # encode image as base 64
	with open(image_path, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode("utf-8")
	
def encode_file(file_path): # encode PDF as base 64
	with open(file_path, "rb") as f:
		data = f.read()
	return base64.b64encode(data).decode("utf-8")