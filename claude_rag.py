#!/usr/bin/env python3
# Requirements: pip install langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu pypdf
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import argparse
import base64
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Set environment variable to avoid parallelism warnings with tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import your existing prompts
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


def process_pdf_for_rag(pdf_path, query=None):
    """
    Process a PDF file for RAG by splitting into chunks and creating embeddings.

    Args:
        pdf_path (str): Path to the PDF file
        query (str): The query to search for (defaults to USER_PROMPT)

    Returns:
        tuple: (vectorstore, contexts) - the FAISS vectorstore and relevant contexts
    """
    if query is None:
        query = USER_PROMPT

    print("Loading PDF...")
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"PDF loaded with {len(documents)} pages.")

    # Use smaller chunks with more overlap for better context retention
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks
        chunk_overlap=300,  # More overlap to maintain context
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Split into {len(chunks)} chunks.")

    # Create embeddings using HuggingFace's embedding model
    print("Initializing embeddings model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"  # This is a good general-purpose embedding model
    )

    print("Creating vector store with FAISS...")
    # Create the vector store
    start_time = time.time()
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    end_time = time.time()

    print(f"Vector store created in {end_time - start_time:.2f} seconds.")

    # Retrieve more chunks to get better context
    k = 8  # Increased from 5
    print(f"Searching for documents relevant to the query: '{query[:50]}...'")
    start_time = time.time()
    similar_docs = vectorstore.similarity_search(query, k=k)
    end_time = time.time()

    print(f"Found {len(similar_docs)} relevant chunks in {end_time - start_time:.2f} seconds.")

    # Display the first few words of each chunk for debugging
    for i, doc in enumerate(similar_docs):
        preview = doc.page_content[:50].replace('\n', ' ').strip()
        print(f"Chunk {i + 1}: {preview}...")

    # Concatenate the chunks into a single context string with clear separators
    context_sections = [f"[SECTION {i + 1}]\n{doc.page_content}" for i, doc in enumerate(similar_docs)]
    context = "\n\n".join(context_sections)

    return context


def main():
    parser = argparse.ArgumentParser(prog="Claude API RAG client")

    parser.add_argument("-i", "--image", required=False, help="path to input image file")
    parser.add_argument("-g", "--guidelines", required=True, help="path to input PDF file for RAG")
    parser.add_argument("-m", "--model", required=True, type=str,
                        choices=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
                                 "claude-3-5-sonnet-20240620", "claude-3-7-sonnet-20250219"],
                        help="Claude model to use")
    parser.add_argument("-q", "--query", required=False, help="Optional custom query for RAG (defaults to USER_PROMPT)")
    args = vars(parser.parse_args())

    # Loading .env file
    load_dotenv()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Warning: ANTHROPIC_API_KEY environment variable not found. Please add it to your .env file or export it.")
        return

    # Use custom query if provided, otherwise use default USER_PROMPT
    query = args.get("query", USER_PROMPT)

    # Process the PDF and get relevant contexts
    relevant_context = process_pdf_for_rag(args["guidelines"], query)

    # Encode PDF/guidelines
    base64_guidelines, guidelines_media_type = encode_file(args["guidelines"])

    # Encode image with proper media type detection if provided
    base64_image = None
    image_media_type = None
    if args["image"]:
        base64_image, image_media_type = encode_file(args["image"])

    client = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    # Enhance the USER_PROMPT with the retrieved context
    rag_prompt = f"""
    {USER_PROMPT}

    Here are some relevant sections from the document that might help answer the question:

    {relevant_context}

    Please analyze the image in relation to the IKEA brand guidelines provided in these sections.
    Identify any violations of the guidelines in the image and provide specific references to the relevant guideline sections.
    Structure your response as a JSON object listing each violation, the guideline reference, and suggested fixes.
    """

    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": []
        }
    ]

    # Add RAG prompt
    messages[0]["content"].append({
        "type": "text",
        "text": rag_prompt
    })

    # Add PDF document
    messages[0]["content"].append({
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": guidelines_media_type,
            "data": base64_guidelines
        }
    })

    # Add image content with correct media type if provided
    if base64_image:
        messages[0]["content"].append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_media_type,
                "data": base64_image
            }
        })

    print("Sending request to Claude API...")
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