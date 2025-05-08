import os
from typing import List, Any, Tuple
import argparse
from dotenv import load_dotenv
import tempfile
import fitz  # PyMuPDF
import json
import hashlib
from datetime import datetime
from openai import OpenAI

# Updated imports based on LlamaIndex documentation
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageNode, TextNode, ImageDocument
from llama_index.core import StorageContext
from llama_index.core import PromptTemplate
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.storage import StorageContext

def parse_arguments():
    parser = argparse.ArgumentParser(description="Multimodal RAG using LlamaIndex and OpenAI GPT-4.1")
    parser.add_argument("--pdf", type=str, required=True, help="Path to input PDF file")
    parser.add_argument("--model", type=str, default="gpt-4.1", choices=["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"], help="OpenAI model to use")
    parser.add_argument("--query", type=str, required=True, help="Query to search for")
    parser.add_argument("--save_images", action="store_true", help="Save extracted images to disk")
    parser.add_argument("--images_dir", type=str, default="./extracted_images", help="Directory to save extracted images")
    parser.add_argument("--index_dir", type=str, default="./saved_index", help="Directory to save/load index")
    parser.add_argument("--rebuild_index", action="store_true", help="Force rebuild the index even if it exists")
    parser.add_argument("--persist_images", action="store_true", help="Keep images after processing (don't delete temp files)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top items to retrieve from index")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for LLM (0.0 for deterministic responses)")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens in the response")
    parser.add_argument("--show_sources", action="store_true", help="Show the sources used to generate the response")
    return parser.parse_args()

def setup_environment():
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return api_key

def generate_index_id(pdf_path: str) -> str:
    """Generate a unique ID for the index based on the PDF file path and modification time."""
    try:
        mod_time = os.path.getmtime(pdf_path)
        file_size = os.path.getsize(pdf_path)
        # Create a hash of the file path + mod time + size to use as the index ID
        index_id = hashlib.md5(f"{pdf_path}_{mod_time}_{file_size}".encode()).hexdigest()
        return index_id
    except Exception as e:
        print(f"Error generating index ID: {e}")
        # Fallback to a simple hash of just the path
        return hashlib.md5(pdf_path.encode()).hexdigest()

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

def setup_llama_index(api_key: str, model_name: str, temperature: float = 0.0, max_tokens: int = 1024):
    # Set up LlamaIndex OpenAI text LLM with deterministic settings
    llm = LlamaOpenAI(
        model=model_name, 
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Set up OpenAI multi-modal model for image processing with deterministic settings
    multi_modal_llm = OpenAIMultiModal(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Configure LlamaIndex settings with text LLM
    Settings.llm = llm
    Settings.chunk_size = 1024
    
    return llm, multi_modal_llm

def build_and_save_index(nodes: List[Any], index_dir: str, index_id: str):
    """Build and save the index to disk."""
    # Create the storage context
    storage_context = StorageContext.from_defaults()
    
    # Create index directory if it doesn't exist
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    
    # Build the index
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    
    # Save the index
    index_path = os.path.join(index_dir, index_id)
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    
    # Save index metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "index_id": index_id,
        "node_count": len(nodes),
        "text_node_count": len([n for n in nodes if isinstance(n, TextNode)]),
        "image_node_count": len([n for n in nodes if isinstance(n, ImageNode)]),
    }
    with open(os.path.join(index_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    
    # Persist the index to disk
    index.storage_context.persist(persist_dir=index_path)
    
    print(f"Index saved to {index_path}")
    return index

def load_index(index_dir: str, index_id: str) -> VectorStoreIndex:
    """Load the index from disk."""
    index_path = os.path.join(index_dir, index_id)
    print(f"Loading index from {index_path}")
    
    try:
        # Load the storage context
        storage_context = StorageContext.from_defaults(
            persist_dir=index_path
        )
        
        # Load the index using the correct method
        # The method name has changed in newer versions of LlamaIndex
        try:
            # Try newer method first
            from llama_index.core import load_index_from_storage
            index = load_index_from_storage(storage_context)
        except (ImportError, AttributeError):
            # Fall back to older method if available
            index = VectorStoreIndex.load_from_disk(
                persist_dir=index_path
            )
            
        return index
    except Exception as e:
        print(f"Error loading index: {e}")
        print("Will rebuild the index instead.")
        return None

def check_if_index_exists(index_dir: str, index_id: str) -> bool:
    """Check if an index already exists for the given PDF."""
    index_path = os.path.join(index_dir, index_id)
    return os.path.exists(index_path)

def query_multimodal_index(index, query: str, multi_modal_llm, top_k: int = 5, show_sources: bool = False):
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
        "Be deterministic in your answer - for the same query and context, you should give the same response.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_tmpl = PromptTemplate(qa_tmpl_str)
    
    # Get retrieved nodes including both text and image nodes with consistent top_k
    retriever = index.as_retriever(similarity_top_k=top_k)
    retrieved_nodes = retriever.retrieve(query)
    
    # Extract image nodes and text nodes
    image_nodes = [node for node in retrieved_nodes if isinstance(node.node, ImageNode)]
    text_nodes = [node for node in retrieved_nodes if not isinstance(node.node, ImageNode)]
    
    # Create context string from text nodes
    context_str = "\n\n".join([node.node.get_content() for node in text_nodes])
    
    # Print source info if requested
    if show_sources:
        print(f"\nSources used for answering the query:")
        for i, node in enumerate(text_nodes):
            print(f"Text {i+1}: {node.node.get_content()[:100]}...")
        for i, node in enumerate(image_nodes):
            print(f"Image {i+1}: {node.node.metadata.get('source', 'Unknown')}")
    
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
    
    # Generate a unique ID for the index based on the PDF file
    index_id = generate_index_id(args.pdf)
    
    # Check if the index already exists
    index_exists = check_if_index_exists(args.index_dir, index_id)
    
    # Setup LlamaIndex - get both text LLM and multi-modal LLM with deterministic settings
    llm, multi_modal_llm = setup_llama_index(
        api_key, 
        args.model, 
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Build or load the index
    if not index_exists or args.rebuild_index:
        print(f"Building new index for {args.pdf}")
        # Process PDF and create nodes
        nodes = create_nodes_from_pdf(args.pdf, args.save_images, args.images_dir)
        
        if not nodes:
            print("No content extracted from the PDF.")
            return
            
        # Build and save the index
        index = build_and_save_index(nodes, args.index_dir, index_id)
        
        # Track the temporary image files to clean up later if needed
        temp_image_nodes = [node for node in nodes if isinstance(node, ImageNode) and not args.save_images]
    else:
        print(f"Loading existing index for {args.pdf}")
        # Load the index from disk
        index = load_index(args.index_dir, index_id)
        
        # If loading failed, build a new index
        if index is None:
            print(f"Building new index for {args.pdf} after failed load")
            nodes = create_nodes_from_pdf(args.pdf, args.save_images, args.images_dir)
            
            if not nodes:
                print("No content extracted from the PDF.")
                return
                
            # Build and save the index
            index = build_and_save_index(nodes, args.index_dir, index_id)
            
            # Track the temporary image files to clean up later if needed
            temp_image_nodes = [node for node in nodes if isinstance(node, ImageNode) and not args.save_images]
        else:
            temp_image_nodes = []  # No temporary files to clean up when loading from disk
    
    # Query the index using the multimodal LLM with consistent parameters
    response = query_multimodal_index(
        index, 
        args.query, 
        multi_modal_llm, 
        top_k=args.top_k,
        show_sources=args.show_sources
    )
    
    # Print response
    print("\nQuery:", args.query)
    print("\nResponse:")
    if hasattr(response, 'response'):
        print(response.response)
    else:
        print(response)
    
    # Clean up temporary files
    if not args.persist_images:
        for node in temp_image_nodes:
            try:
                if os.path.exists(node.image_path):
                    os.unlink(node.image_path)
            except Exception as e:
                print(f"Error removing temporary file {node.image_path}: {e}")

if __name__ == "__main__":
    run_multimodal_rag() 