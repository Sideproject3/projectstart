# --- Imports ---
from llama_parse import LlamaParse
from llama_index.core import KnowledgeGraphIndex, StorageContext, SimpleDirectoryReader
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from PIL import Image
from dotenv import load_dotenv
import os 

load_dotenv()

# --- Step 1: Parse PDF Guidelines (with images) ---
parser = LlamaParse(
	api_key=os.environ.get("LLAMA_CLOUD_API_KEY"),
	use_vendor_multimodal_model=True,
	vendor_multimodal_model_name="openai-gpt-4-1",  # Or your preferred model
	result_type="markdown"
)

documents = parser.load_data("ikea.pdf")

# --- Step 2: Build Knowledge Graph Index ---
embed_model = OpenAIEmbedding()
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

kg_index = KnowledgeGraphIndex.from_documents(
	documents,
	max_triplets_per_chunk=5,
	storage_context=storage_context,
	embed_model=embed_model,
	include_embeddings=True
)

# --- Step 3: Define the Compliance Check Function ---
def check_image_against_guidelines(user_query):
	# Retrieve relevant guideline nodes for the query
	retriever = kg_index.as_retriever()
	relevant_nodes = retriever.retrieve(user_query)
	guideline_text = "\n\n".join([node.text for node in relevant_nodes])

	# Load the uploaded image
	image_documents = SimpleDirectoryReader("./ikea_images").load_data()

	# Initialize the multimodal LLM (GPT-4 Vision)
	mm_llm = OpenAIMultiModal(model="gpt-4o")

	# Create the prompt
	prompt = f"""
	The following are extracted guidelines:
	{guideline_text}

	Does the attached image violate any of these guidelines? Please answer YES or NO and provide a short explanation.
	"""

	# Query the multimodal model
	response = mm_llm.complete(
		prompt=prompt,
		image_documents=image_documents
	)
	return response

# --- Step 4: Use the System ---
if __name__ == "__main__":
	# Example usage
	user_query = "use the provided guidelines which are provided as a PDF file and reflect whether the image provided violates any of the guidelines"

	result = check_image_against_guidelines(user_query)
	print(result)
