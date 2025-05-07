import os
from llama_index.core import (
	VectorStoreIndex,
	KnowledgeGraphIndex,
	StorageContext,
	Settings,
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import ImageNode
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_parse import LlamaParse
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import fitz  # PyMuPDF
from dotenv import load_dotenv

class RobustPDFToGraphRAG:
	def __init__(self, pdf_path, storage_dir="graph_rag_storage"):
		self.pdf_path = pdf_path
		self.storage_dir = storage_dir
		os.makedirs(self.storage_dir, exist_ok=True)

		load_dotenv()

		# Initialize components
		self._initialize_components()
        
	def _initialize_components(self):
		"""Initialize all required components with error handling"""
		try:
			self.parser = LlamaParse(
				api_key=os.environ.get("LLAMA_CLOUD_API_KEY"),
				result_type="markdown",
				parsing_instruction="Extract all content with structure preserved",
				max_timeout=600,
			)
            
			Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
			Settings.llm = OpenAI(model="gpt-4-turbo")
			Settings.multi_modal_llm = OpenAIMultiModal(
				model="gpt-4-vision-preview", 
				max_new_tokens=1000
			)
			Settings.node_parser = SemanticSplitterNodeParser(
				buffer_size=1,
				breakpoint_percentile_threshold=95,
				embed_model=Settings.embed_model
			)
		except Exception as e:
			raise RuntimeError(f"Failed to initialize components: {str(e)}")

	def process_pdf(self):
		"""Process PDF with error handling"""
		try:
			# Parse with LlamaParse
			documents = self.parser.load_data(self.pdf_path)
            
			# Extract images
			image_nodes = []
			doc = fitz.open(self.pdf_path)
			for page_num in range(len(doc)):
				page = doc.load_page(page_num)
				text = page.get_text()
                
				for img_index, img in enumerate(page.get_images(full=True)):
					xref = img[0]
					base_image = doc.extract_image(xref)
					image_path = f"{self.storage_dir}/img_{page_num}_{img_index}.png"
                    
					with open(image_path, "wb") as f:
						f.write(base_image["image"])
                    
					image_nodes.append(
						ImageNode(
							image=image_path,
							metadata={
								"page": page_num + 1,
								"text_context": text[:1000]
							}
						)
					)
			doc.close()
			return documents, image_nodes
		except Exception as e:
			raise RuntimeError(f"Failed to process PDF: {str(e)}")

	def build_indices(self):
		"""Build new indices with proper storage handling"""
		try:
			# Process the PDF
			documents, image_nodes = self.process_pdf()
            
			# Create fresh storage contexts
			graph_storage = StorageContext.from_defaults(
				graph_store=SimpleGraphStore(),
				persist_dir=os.path.join(self.storage_dir, "graph")
			)
            
			mm_storage = StorageContext.from_defaults(
				persist_dir=os.path.join(self.storage_dir, "multimodal")
			)
            
			# Build Knowledge Graph Index
			kg_index = KnowledgeGraphIndex.from_documents(
				documents,
				storage_context=graph_storage,
				max_triplets_per_chunk=5,
				include_embeddings=True,
				show_progress=True
			)
            
			# Build Multimodal Index
			mm_index = MultiModalVectorStoreIndex(
				image_nodes,
				storage_context=mm_storage
			)
            
			# Persist indices
			kg_index.storage_context.persist()
			mm_index.storage_context.persist()
            
			return kg_index, mm_index
		except Exception as e:
			raise RuntimeError(f"Failed to build indices: {str(e)}")

	def load_indices(self):
		"""Load indices with proper error handling"""
		try:
			# Check if storage exists
			if not os.path.exists(os.path.join(self.storage_dir, "graph")):
				raise FileNotFoundError("No stored indices found - please build them first")
            
			# Load indices
			kg_index = KnowledgeGraphIndex.from_persist_dir(
				persist_dir=os.path.join(self.storage_dir, "graph")
			)
            
			mm_index = MultiModalVectorStoreIndex.from_persist_dir(
				persist_dir=os.path.join(self.storage_dir, "multimodal")
			)
            
			return kg_index, mm_index
		except Exception as e:
			raise RuntimeError(f"Failed to load indices: {str(e)}")

	def query(self, query_text, kg_index=None, mm_index=None):
		"""Safe query execution"""
		try:
			if kg_index is None or mm_index is None:
				kg_index, mm_index = self.load_indices()
            
			# Create query engines
			kg_engine = kg_index.as_query_engine(
				include_text=True,
				response_mode="tree_summarize"
			)
            
			mm_engine = mm_index.as_query_engine()
            
			# Execute queries
			text_response = kg_engine.query(query_text)
			image_response = mm_engine.query(query_text)
            
			return {
				"text": str(text_response),
				"images": [
					node.metadata["image_path"]
					for node in getattr(image_response, "source_nodes", [])
					if hasattr(node, "metadata") and "image_path" in node.metadata
				]
			}
		except Exception as e:
			raise RuntimeError(f"Query failed: {str(e)}")
	
# First run (or when you need to rebuild)
processor = RobustPDFToGraphRAG("ikea.pdf")
kg_index, mm_index = processor.build_indices()  # Creates new indices

# Subsequent runs
# processor = RobustPDFToGraphRAG("your_file.pdf")
# try:
# 	kg_index, mm_index = processor.load_indices()  # Loads existing
# except FileNotFoundError:
# 	kg_index, mm_index = processor.build_indices()  # Fallback to build

# Querying
query = "what does this guideline tell us that applies to specifially logos ?"

response = processor.query(query)

print(response["text"])

for img_path in response["images"]:
	print(f"Relevant image: {img_path}")





