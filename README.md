# Multimodal RAG with LlamaIndex and OpenAI

This application demonstrates how to implement Multimodal Retrieval Augmented Generation (RAG) using LlamaIndex and OpenAI's GPT-4.1 model. It processes a single PDF file, extracting both text and images, and allows you to query the content using natural language.

## Requirements

- Python 3.8+
- OpenAI API key

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the application with a PDF file and query:

```bash
python multimodal_rag.py --pdf your_document.pdf --query "Your question here"
```

### Optional Arguments

- `--model`: OpenAI model to use (default: "gpt-4.1", choices: ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"])
- `--save_images`: Save extracted images to disk (default: images are stored in temporary files)
- `--images_dir`: Directory to save extracted images (default: "./extracted_images")

Example with all options:

```bash
python multimodal_rag.py --pdf your_document.pdf --model gpt-4.1 --save_images --images_dir ./my_images --query "Summarize the content of this document"
```

## How It Works

1. The application extracts text content from the PDF file
2. It identifies and extracts images embedded in the PDF
3. The text and images are processed into nodes for indexing
4. LlamaIndex is used to build a vector index of the content
5. The query is processed against the index using OpenAI's multimodal capabilities
6. Results are returned based on both the text and visual content of your PDF

## Features

- Processes both text and images from a single PDF document
- Creates temporary image files by default (cleaned up automatically)
- Option to save extracted images for inspection
- Supports multiple OpenAI models for different performance/cost tradeoffs

## Limitations

- Requires an OpenAI API key with access to GPT-4.1 or GPT-4o models
- Image analysis is limited by the capabilities of the OpenAI model
- Very large PDFs may encounter processing limits 