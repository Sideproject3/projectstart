#!/usr/bin/env python3
# Requirements: pip install langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu pypdf google-generativeai
from dotenv import load_dotenv
import os
import argparse
import time
import base64
import google.generativeai as genai  # Using the direct import as in your original code
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Set environment variable to avoid parallelism warnings with tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import your existing prompts
from prompts import SYSTEM_MESSAGE, USER_PROMPT


def process_pdf_for_rag(pdf_path, query=None):
    """
    Process a PDF file for RAG by splitting into chunks and creating embeddings.

    Args:
        pdf_path (str): Path to the PDF file
        query (str): The query to search for (defaults to USER_PROMPT)

    Returns:
        str: relevant contexts from the PDF for RAG
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


def load_file_as_base64(file_path):
    """
    Load and encode a file as base64.

    Args:
        file_path (str): Path to the file

    Returns:
        str: Base64 encoded file content
    """
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(prog="Gemini API RAG client")

    parser.add_argument("-i", "--image", required=False, help="path to input image file")
    parser.add_argument("-g", "--guidelines", required=True, help="path to input PDF file for RAG")
    parser.add_argument("-m", "--model", required=False, type=str,
                        choices=["gemini-1.0-pro", "gemini-1.5-pro", "gemini-1.5-flash",
                                 "gemini-2.5-flash-preview-04-17", "gemini-2.0-flash",
                                 "gemini-2.5-pro-preview-03-25", "gemini-2.5-pro-exp-03-25"],
                        default="gemini-1.5-pro",
                        help="Gemini model to use")
    parser.add_argument("-q", "--query", required=False, help="Optional custom query for RAG (defaults to USER_PROMPT)")
    args = vars(parser.parse_args())

    # Loading .env file
    load_dotenv()

    # Check for Gemini API key
    if not os.environ.get("GEMINI_API_KEY"):
        print(
            "Warning: GEMINI_API_KEY environment variable not found. Please add it to your .env file or export it.")
        return

    # Configure the Gemini API with your API key
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

    # Use custom query if provided, otherwise use default USER_PROMPT
    query = args.get("query", USER_PROMPT)

    # Process the PDF and get relevant contexts
    relevant_context = process_pdf_for_rag(args["guidelines"], query)

    # Prepare the model
    model = genai.GenerativeModel(args["model"])

    # Enhance the USER_PROMPT with the retrieved context
    rag_prompt = f"""
    {SYSTEM_MESSAGE}

    {USER_PROMPT}

    Here are some relevant sections from the document that might help answer the question:

    {relevant_context}

    Please analyze the image in relation to the IKEA brand guidelines provided in these sections.
    Identify any violations of the guidelines in the image and provide specific references to the relevant guideline sections.
    Structure your response as a JSON object listing each violation, the guideline reference, and suggested fixes.
    """

    # Prepare the request
    content = [rag_prompt]

    # Add image if provided
    if args["image"]:
        # Read the image file
        with open(args["image"], "rb") as f:
            image_data = f.read()

        # Add the image to the content parts
        content = [{
            "text": rag_prompt
        }, {
            "inline_data": {
                "mime_type": "image/jpeg",  # You may need to adjust this based on the image type
                "data": base64.b64encode(image_data).decode("utf-8")
            }
        }]

    print("Sending request to Gemini API...")
    try:
        # Generate the response
        response = model.generate_content(
            content,
            generation_config={
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 4096,
            }
        )

        # Print the response
        print(response.text)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()