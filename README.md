# AI-Powered Personal Search and Metadata Extraction Tool (In-Progress)

## Motivation
If you're anything like me, your computer is full of python scripts and jupyter notebooks from courses, projects, and repos accumulated over the years. I have TERRIBLE memory when it comes to this kind of stuff, often forgetting about useful code I've used or developed over the years. Rather than spending wasted time endlessly searching local directories for code, I wanted a tool to help me catalog my previous work, create rich metadata about it's contents, and allow me to search smarter so I can quickly leverage that work for my current projects. I'm also using this as an opportunity to improve my personal productivity using AI tools like Cursor and Windsurf.


## Features (Current Functionality)
- This tool generates and extracts metadata from Python scripts (`.py`) and Jupyter notebooks (`.ipynb`) on my local machine.
- Extracts file name, path, last modified date
- Auto-generates a summary and topic tags using Anthropic Claude API
- Lists libraries and functions used in the code
- Generates an embedding from the code contents & metdadata using Huggingface Sentence Transformers embedding model
- Appends metadata to a growing JSON file
- Handles both single files and directories
- Modular, robust, and PEP8-compliant


## Major Enhancements (In-Progress)
- Implement batch processing for large codebases. I want all the python code on my computer processed, need to figure out the best way to do it.
- Store outputs in a vector database for storage, fast rerieval, and ANN search
- Frontend Tool: Streamlit App for the frontend search interface
- Leverage embeddings for file similarity scoring. Given a file I'm interested in quickly accessing other similar projects I've worked on
- Semantic Search: search codebase using natural language and content embeddings
- LLM + RAG Chatbot: Search, interact, and enhance my codebase using an LLM which has knowledge of all my files
- Code Enhancement Agent: Develop an agent that'll proacively guide me through the process of enhancing and optimizing my codebase based my learning interests and development areas



## Usage
```
pipenv run python notebook_metadata_logging.py
```

## Setup
1. Create a `.env` file with your Anthropic API key and Huggingface API key:
   ```
   ANTHROPIC_API_KEY=your_key_here
   HUGGINGFACE_API_KEY=your_api_key
   ```
2. Install dependencies:
   ```
   pipenv install
   ```

## Output Example
See `src/output/output.json` for output format.

## License
MIT
