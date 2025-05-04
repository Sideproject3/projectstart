#!/usr/bin/env python3
# !/usr/bin/env python3
# Requirements: pip install langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu pypdf
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import argparse
import base64
import tempfile
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import time

# Set environment variable to avoid parallelism warnings with tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


def create_vector_store(pdf_path):
    """
    Creates a vector store from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        vectorstore: The vector store with the embedded PDF chunks
    """
    print("Loading PDF...")
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"PDF loaded with {len(documents)} pages.")

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
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

    return vectorstore


def retrieve_relevant_context(vectorstore, query, k=5):
    """
    Retrieves the most relevant context for a given query.

    Args:
        vectorstore: The vector store to search in
        query (str): The query to search for
        k (int): Number of chunks to retrieve

    Returns:
        str: The concatenated relevant text chunks
    """
    # Search for similar chunks
    print(f"Searching for documents relevant to the query: '{query[:50]}...'")
    start_time = time.time()
    similar_docs = vectorstore.similarity_search(query, k=k)
    end_time = time.time()

    print(f"Found {len(similar_docs)} relevant chunks in {end_time - start_time:.2f} seconds.")

    # Concatenate the chunks into a single context string
    context = "\n\n".join([doc.page_content for doc in similar_docs])

    return context


def main():
    parser = argparse.ArgumentParser(prog="Claude API RAG client")

    parser.add_argument("-i", "--image", required=False, help="path to input image file")
    parser.add_argument("-g", "--guidelines", required=True, help="path to input PDF file for RAG")
    parser.add_argument("-m", "--model", required=True, type=str,
                        choices=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
                                 "claude-3-5-sonnet-20240620", "claude-3-7-sonnet-20250219"],
                        help="Claude model to use")
    args = vars(parser.parse_args())

    # loading .env file
    load_dotenv()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Warning: ANTHROPIC_API_KEY environment variable not found. Please add it to your .env file or export it.")
        return

    # Create vector store from the PDF (using guidelines as the PDF for RAG)
    vectorstore = create_vector_store(args["guidelines"])

    # Retrieve relevant context based on the USER_PROMPT
    relevant_context = retrieve_relevant_context(vectorstore, USER_PROMPT)

    # Create vector store from the PDF (using guidelines as the PDF for RAG)
    print("Creating vector store from PDF...")
    vectorstore = create_vector_store(args["guidelines"])

    # Retrieve relevant context based on the USER_PROMPT
    print("Retrieving relevant context...")
    relevant_context = retrieve_relevant_context(vectorstore, USER_PROMPT)

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

    Please answer based on both the image and the provided context from the document.
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

    # Cleanup the vector database
    import shutil
    shutil.rmtree(vectorstore._persist_directory, ignore_errors=True)


if __name__ == "__main__":
    main()