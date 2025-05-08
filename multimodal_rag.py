import os
from typing import List, Dict, Any, Tuple
import argparse
from dotenv import load_dotenv
import tempfile
import fitz  # PyMuPDF
from PIL import Image
import io
import uuid
from openai import OpenAI

# Updated imports based on LlamaIndex documentation
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageNode, TextNode, ImageDocument
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import PromptTemplate
from llama_index.core.response import Response
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.query_engine import SimpleMultiModalQueryEngine

def parse_arguments():
    parser = argparse.ArgumentParser(description="Multimodal RAG using LlamaIndex and OpenAI GPT-4.1")
    parser.add_argument("--pdf", type=str, required=True, help="Path to input PDF file")
    parser.add_argument("--model", type=str, default="gpt-4.1", choices=["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"], help="OpenAI model to use")
    parser.add_argument("--query", type=str, required=True, help="Query to search for")
    parser.add_argument("--save_images", action="store_true", help="Save extracted images to disk")
    parser.add_argument("--images_dir", type=str, default="./extracted_images", help="Directory to save extracted images")
    return parser.parse_args()

def setup_environment():
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return api_key

def extract_images_from_pdf(pdf_path: str, save_images: bool = False, images_dir: str = "./extracted_images") -> List[Tuple[str, bytes]]:
    """Extract images from PDF file and optionally save them to disk."""
    if save_images and not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    image_list = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            # Get images
            image_list_for_page = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list_for_page):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Create a unique filename for each image
                image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{base_image['ext']}"
                
                # Save image if requested
                if save_images:
                    image_path = os.path.join(images_dir, image_filename)
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    print(f"Saved image: {image_path}")
                    image_list.append((image_path, image_bytes))
                else:
                    # Create temporary file to use with LlamaIndex
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{base_image['ext']}")
                    temp_file.write(image_bytes)
                    temp_file.close()
                    image_list.append((temp_file.name, image_bytes))
        
        print(f"Extracted {len(image_list)} images from PDF")
        return image_list
    
    except Exception as e:
        print(f"Error extracting images: {e}")
        return []

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def create_nodes_from_pdf(pdf_path: str, save_images: bool = False, images_dir: str = "./extracted_images") -> List[Any]:
    """Create text and image nodes from a PDF file."""
    nodes = []
    
    # Extract text
    text_content = extract_text_from_pdf(pdf_path)
    if text_content:
        # Create document from text
        text_doc = Document(text=text_content, metadata={"source": pdf_path, "content_type": "text"})
        
        # Split into nodes
        splitter = SentenceSplitter(chunk_size=1024)
        text_nodes = splitter.get_nodes_from_documents([text_doc])
        nodes.extend(text_nodes)
        print(f"Created {len(text_nodes)} text nodes from PDF")
    
    # Extract images
    images = extract_images_from_pdf(pdf_path, save_images, images_dir)
    for img_path, img_bytes in images:
        image_node = ImageNode(
            image_path=img_path,
            image_data=img_bytes,
            metadata={"source": pdf_path, "content_type": "image"}
        )
        nodes.append(image_node)
    
    return nodes

def setup_llama_index(api_key: str, model_name: str):
    # Set up LlamaIndex OpenAI text LLM
    llm = LlamaOpenAI(model=model_name, api_key=api_key)
    
    # Set up OpenAI multi-modal model for image processing
    multi_modal_llm = OpenAIMultiModal(
        model=model_name,
        api_key=api_key,
    )
    
    # Configure LlamaIndex settings with text LLM
    Settings.llm = llm
    Settings.chunk_size = 1024
    
    return llm, multi_modal_llm

def build_index(nodes: List[Any]):
    # Create vector store index
    index = VectorStoreIndex(nodes)
    return index

def query_multimodal_index(index, query: str, multi_modal_llm):
    """Query the index using a multimodal approach that can process both text and images."""
    
    # Create custom prompt template for better results
    qa_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information including both text and images, "
        "please provide a comprehensive answer to the query.\n"
        "Analyze both the textual content and any images to give a complete response.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_tmpl = PromptTemplate(qa_tmpl_str)
    
    # Get retrieved nodes including both text and image nodes
    retriever = index.as_retriever(similarity_top_k=5)
    retrieved_nodes = retriever.retrieve(query)
    
    # Extract image nodes and text nodes
    image_nodes = [node for node in retrieved_nodes if isinstance(node.node, ImageNode)]
    text_nodes = [node for node in retrieved_nodes if not isinstance(node.node, ImageNode)]
    
    # Create context string from text nodes
    context_str = "\n\n".join([node.node.get_content() for node in text_nodes])
    
    print(f"Retrieved {len(text_nodes)} text nodes and {len(image_nodes)} image nodes for query")
    
    # If we have images, process them with the multimodal LLM
    if image_nodes:
        # Create proper ImageDocument objects for multimodal processing
        image_documents = []
        for node in image_nodes:
            try:
                image_node = node.node
                if hasattr(image_node, 'image_path') and os.path.exists(image_node.image_path):
                    # Create an ImageDocument with the image data
                    if hasattr(image_node, 'image_data') and image_node.image_data:
                        # Use stored image data if available
                        image_documents.append(ImageDocument(image=image_node.image_data))
                    else:
                        # Read from file if needed
                        with open(image_node.image_path, "rb") as f:
                            image_bytes = f.read()
                            image_documents.append(ImageDocument(image=image_bytes))
            except Exception as e:
                print(f"Error loading image from node: {e}")
        
        # Format the prompt with context
        formatted_prompt = qa_tmpl.format(context_str=context_str, query_str=query)
        
        # Use multimodal LLM to get response considering both text and images
        try:
            response = multi_modal_llm.complete(
                prompt=formatted_prompt,
                image_documents=image_documents
            )
            return response
        except Exception as e:
            print(f"Error during multimodal processing: {e}")
            print("Falling back to text-only processing")
            # Fall back to text-only processing
            query_engine = index.as_query_engine(
                text_qa_template=qa_tmpl
            )
            response = query_engine.query(query)
            return response
    else:
        # If no images, use regular query engine with text
        query_engine = index.as_query_engine(
            text_qa_template=qa_tmpl
        )
        response = query_engine.query(query)
        return response

def run_multimodal_rag():
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment
    api_key = setup_environment()
    
    # Validate PDF path
    if not os.path.exists(args.pdf):
        print(f"Error: PDF file not found at {args.pdf}")
        return
    
    # Process PDF and create nodes
    nodes = create_nodes_from_pdf(args.pdf, args.save_images, args.images_dir)
    
    if not nodes:
        print("No content extracted from the PDF.")
        return
    
    # Setup LlamaIndex - get both text LLM and multi-modal LLM
    llm, multi_modal_llm = setup_llama_index(api_key, args.model)
    
    # Build index
    index = build_index(nodes)
    
    # Query the index using the multimodal LLM
    response = query_multimodal_index(index, args.query, multi_modal_llm)
    
    # Print response
    print("\nQuery:", args.query)
    print("\nResponse:")
    if hasattr(response, 'response'):
        print(response.response)
    else:
        print(response)
    
    # Clean up temporary files
    for node in nodes:
        if isinstance(node, ImageNode) and not args.save_images:
            try:
                if os.path.exists(node.image_path):
                    os.unlink(node.image_path)
            except Exception as e:
                print(f"Error removing temporary file {node.image_path}: {e}")

if __name__ == "__main__":
    run_multimodal_rag() 