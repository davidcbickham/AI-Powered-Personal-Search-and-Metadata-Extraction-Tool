import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from transformers import AutoModel
from dynaconf import Dynaconf

# External dependencies
import nbformat
import anthropic

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# --- Configuration ---
# Get the project root directory (where this script is located)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Initialize dynaconf settings
settings = Dynaconf(
    root_path=PROJECT_ROOT,
    settings_files=['settings.yaml'],
    environments=True,
    env='default',
    envvar_prefix=False,
)


# Paths from config (resolved relative to project root)
INPUT_PATH = str(PROJECT_ROOT / settings.INPUT_PATH)
JSON_PATH = str(PROJECT_ROOT / settings.JSON_PATH)
ENV_PATH = str(PROJECT_ROOT / settings.ENV_PATH)

load_dotenv(dotenv_path=ENV_PATH)
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

if not ANTHROPIC_API_KEY:
    logging.error("ANTHROPIC_API_KEY not found in environment variables.")
    sys.exit(1)

if not HUGGINGFACE_API_KEY:
    logging.error("HUGGINGFACE_API_KEY not found in environment variables.")
    sys.exit(1)

# --- Embedding Model Cache ---
_embedding_model_cache = None
_embedding_model_name = None
_embedding_api_key = None

# Anthropic API client setup
def get_anthropic_client():
    """
    Initialize and return an Anthropic API client using the API key from environment variables.
    Returns:
        anthropic.Anthropic: Anthropic API client instance.
    """
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# --- Utility Functions ---
def get_last_modified(file_path: str) -> str:
    """
    Get the last modified date of a file as a string in YYYY-MM-DD format.
    Args:
        file_path (str): Path to the file.
    Returns:
        str: Last modified date as 'YYYY-MM-DD', or empty string on error.
    """
    try:
        timestamp = os.path.getmtime(file_path)
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime('%Y-%m-%d')
    except Exception as e:
        logging.error(f"Error getting last modified date for {file_path}: {e}")
        return ""

def extract_imports_py(file_content: str) -> List[str]:
    """
    Extract imported symbols from Python source code.
    Args:
        file_content (str): Python file content as a string.
    Returns:
        List[str]: List of imported module/class/function names.
    """
    import ast
    try:
        tree = ast.parse(file_content)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    # For 'import x as y', n.name is 'x'
                    imports.add(n.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                # For 'from x import y, z', add 'y', 'z', etc.
                for n in node.names:
                    imports.add(n.name)
        return sorted(list(imports))
    except Exception as e:
        logging.error(f"Error extracting imports: {e}")
        return []

def extract_functions_py(file_content: str) -> List[str]:
    """
    Extract function names defined in a Python source file.
    Args:
        file_content (str): Python file content as a string.
    Returns:
        List[str]: List of function names defined in the file.
    """
    import ast
    try:
        tree = ast.parse(file_content)
        return sorted([node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
    except Exception as e:
        logging.error(f"Error extracting functions: {e}")
        return []

def extract_imports_ipynb(nb_node: nbformat.NotebookNode) -> List[str]:
    """
    Extract imported symbols from all code cells in a Jupyter notebook.
    Args:
        nb_node (nbformat.NotebookNode): Parsed notebook object.
    Returns:
        List[str]: List of imported module/class/function names from code cells.
    """
    imports = set()
    try:
        for cell in nb_node.cells:
            if cell.cell_type == 'code':
                imports.update(extract_imports_py(cell.source))
        return sorted(list(imports))
    except Exception as e:
        logging.error(f"Error extracting notebook imports: {e}")
        return []

def extract_functions_ipynb(nb_node: nbformat.NotebookNode) -> List[str]:
    """
    Extract function names defined in all code cells of a Jupyter notebook.
    Args:
        nb_node (nbformat.NotebookNode): Parsed notebook object.
    Returns:
        List[str]: List of function names defined in code cells.
    """
    functions = set()
    try:
        for cell in nb_node.cells:
            if cell.cell_type == 'code':
                functions.update(extract_functions_py(cell.source))
        return sorted(list(functions))
    except Exception as e:
        logging.error(f"Error extracting notebook functions: {e}")
        return []

# --- Anthropic API ---
SUMMARY_PROMPT_TEMPLATE = (
    "You are a technical assistant. Read the following Python or Jupyter notebook content "
    "and generate a 3–5 sentence paragraph summarizing what the script or notebook does. "
    "Ensure the summary is written concisely and efficiently. "
    "Focus on key steps like data loading, preprocessing, modeling, and evaluation.\n\n"
    "Content:\n{content}"
)

TOPIC_PROMPT_TEMPLATE = (
    "You are a machine learning engineer assistant.\n\n"
    "You will be shown the content of a Python script or Jupyter notebook. Your task is to infer a set of high-level machine learning, data science, or domain-relevant topics that the code is associated with.\n\n"
    "Return a single comma-separated string of 2–5 topic names. Topics should be concise, abstracted, and suitable for tagging or categorization purposes.\n\n"
    "Consider topics such as:\n"
    "- ML subfields (e.g., Supervised Learning, NLP, Computer Vision)\n"
    "- Techniques (e.g., Clustering, Dimensionality Reduction, Transfer Learning)\n"
    "- Application areas (e.g., Recommendation Systems, Time Series Forecasting, Fraud Detection)\n"
    "- Relevant libraries or frameworks (e.g., PyTorch, Scikit-learn, HuggingFace)\n\n"
    "Base your inference only on content clearly demonstrated in the source.\n\n"
)

def call_anthropic(prompt: str, client, max_tokens=None) -> str:
    """
    Call the Anthropic API with a given prompt and return the response text.
    Args:
        prompt (str): Prompt string to send to the model.
        client: Anthropic API client instance.
        max_tokens (int): Maximum tokens for the response (uses config default if None).
    Returns:
        str: Response text from the API, or empty string on error.
    """
    try:
        if max_tokens is None:
            max_tokens = settings.MAX_TOKENS_SUMMARY
        response = client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            temperature=settings.TEMPERATURE,
            system="You are a metadata extraction assistant.",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        logging.error(f"Anthropic API error: {e}")
        return ""

def get_summary(file_content: str, client) -> str:
    """
    Generate a summary for the file content using the Anthropic API.
    Args:
        file_content (str): Source code or notebook content.
        client: Anthropic API client instance.
    Returns:
        str: Concise summary generated by the model.
    """
    truncation_limit = settings.CONTENT_TRUNCATION_LIMIT
    prompt = SUMMARY_PROMPT_TEMPLATE.format(content=file_content[:truncation_limit])
    summary = call_anthropic(prompt, client, max_tokens=settings.MAX_TOKENS_SUMMARY)
    return summary

def get_topics(file_content: str, client) -> list:
    """
    Generate a list of high-level topics for the file content using the Anthropic API.
    Args:
        file_content (str): Source code or notebook content.
        client: Anthropic API client instance.
    Returns:
        list: List of topic strings generated by the model.
    """
    import re
    truncation_limit = settings.CONTENT_TRUNCATION_LIMIT
    prompt = TOPIC_PROMPT_TEMPLATE + f"Content:\n{file_content[:truncation_limit]}"
    topics_str = call_anthropic(prompt, client, max_tokens=settings.MAX_TOKENS_TOPICS)
    # Parse comma-separated topics, strip whitespace
    topics = [topic.strip() for topic in topics_str.split(',') if topic.strip()]
    return topics


def extract_text_from_list(metadata_list):
    """
    Safely convert a list of metadata items (e.g., topics, functions, libraries) into a comma-separated string.
    Args:
        metadata_list (list): List of metadata strings.
    Returns:
        str: Comma-separated string of items, or empty string if input is None/empty.
    """
    try:
        if metadata_list:
            return ", ".join(metadata_list) + "."
        else:
            return ""
    except Exception as e:
        logging.error(f"Error in extract_text_from_list: {e}")
        return ""

def get_embedding_text(summary, *metadata_lists):
    """
    Combine summary and other metadata lists into a single string for embedding generation.
    Args:
        summary (str): The summary text.
        *metadata_lists: Variable number of metadata lists (e.g., topics, functions, libraries).
    Returns:
        str: Concatenated string for embedding.
    """
    try:
        if not metadata_lists:
            return summary
        extra_texts = [extract_text_from_list(lst) for lst in metadata_lists if lst]
        embedding_text = " ".join([summary] + extra_texts)
        return embedding_text
    except Exception as e:
        logging.error(f"Error in get_embedding_text: {e}")
        return summary

def get_embedding_model(embedding_model=None, api_key=None):
    """
    Get or initialize the cached embedding model. Loads the model only once.
    Args:
        embedding_model (str): Model name or path (uses config default if None).
        api_key (str): HuggingFace API key.
    Returns:
        The cached model instance, or None on error.
    """
    global _embedding_model_cache, _embedding_model_name, _embedding_api_key
    
    # Use default from config if not provided
    if embedding_model is None:
        embedding_model = settings.EMBEDDING_MODEL
    
    # Use default API key if not provided
    if api_key is None:
        api_key = HUGGINGFACE_API_KEY
    
    # Return cached model if it matches the requested model and API key
    if (_embedding_model_cache is not None and 
        _embedding_model_name == embedding_model and 
        _embedding_api_key == api_key):
        return _embedding_model_cache
    
    # Load the model if not cached or if model/api_key changed
    try:
        logging.info(f"Loading embedding model: {embedding_model}")
        _embedding_model_cache = AutoModel.from_pretrained(
            embedding_model,
            trust_remote_code=True,
            use_auth_token=api_key
        )
        _embedding_model_name = embedding_model
        _embedding_api_key = api_key
        logging.info("Embedding model loaded successfully.")
        return _embedding_model_cache
    except Exception as e:
        logging.error(f"Error loading embedding model: {e}")
        return None

def get_metadata_embedding(text: str, embedding_model=None, api_key=None):
    """
    Generate a numeric embedding vector from text using a HuggingFace-compatible model.
    The model is cached after first load for efficiency.
    Args:
        text (str): Input text to embed.
        embedding_model (str): Model name or path (uses config default if None).
        api_key (str): HuggingFace API key (optional, uses default if not provided).
    Returns:
        list: List of floats representing the embedding, or empty list on failure.
    """
    try:
        model = get_embedding_model(embedding_model, api_key)
        if model is None:
            return []
        
        embedding = model.encode([text])
        embedding = [float(x) for x in embedding[0, :]]
        return embedding
    except Exception as e:
        logging.error(f"Error in get_metadata_embedding: {e}")
        return []


# --- File Processing Logic ---
def process_py_file(file_path: str, client) -> Dict[str, Any]:
    """
    Process a Python script to extract metadata including summary, topics, imports, and functions.
    Args:
        file_path (str): Path to the .py file.
        client: Anthropic API client instance.
    Returns:
        Dict[str, Any]: Metadata dictionary for the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        summary = get_summary(content, client)
        topics = get_topics(content, client)
        libraries = extract_imports_py(content)
        functions = extract_functions_py(content)
        embedding_text = get_embedding_text(summary, topics, functions, libraries)
        embedding = get_metadata_embedding(embedding_text)
        
        return {
            "File Name": os.path.basename(file_path),
            "File Path": os.path.abspath(file_path),
            "Last Modified": get_last_modified(file_path),
            "Summary": summary,
            "Libraries": libraries,
            "Functions": functions,
            "Topics": topics,
            "Embedding": embedding
        }
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return {}

def process_ipynb_file(file_path: str, client) -> Dict[str, Any]:
    """
    Process a Jupyter notebook to extract metadata including summary, topics, imports, and functions.
    Args:
        file_path (str): Path to the .ipynb file.
        client: Anthropic API client instance.
    Returns:
        Dict[str, Any]: Metadata dictionary for the notebook.
    """
    try:
        nb = nbformat.read(file_path, as_version=4)
        # Concatenate both markdown and code cells for context
        all_text = '\n'.join(
            cell.source for cell in nb.cells if cell.cell_type in ('code', 'markdown')
        )

        summary = get_summary(all_text, client)
        topics = get_topics(all_text, client)
        libraries = extract_imports_ipynb(nb)
        functions = extract_functions_ipynb(nb)
        embedding_text = get_embedding_text(summary, topics, functions, libraries)
        embedding = get_metadata_embedding(embedding_text)
        
        return {
            "File Name": os.path.basename(file_path),
            "File Path": os.path.abspath(file_path),
            "Last Modified": get_last_modified(file_path),
            "Summary": summary,
            "Libraries": libraries,
            "Functions": functions,
            "Topics": topics,
            "Embedding": embedding
        }
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return {}

def process_file(file_path: str, client) -> Dict[str, Any]:
    """
    Process a file (.py or .ipynb) and extract metadata using the appropriate handler.
    Args:
        file_path (str): Path to the file.
        client: Anthropic API client instance.
    Returns:
        Dict[str, Any]: Metadata dictionary for the file or notebook.
    """
    if file_path.endswith('.py'):
        return process_py_file(file_path, client)
    elif file_path.endswith('.ipynb'):
        return process_ipynb_file(file_path, client)
    else:
        logging.warning(f"Unsupported file type: {file_path}")
        return {}

def collect_files(INPUT_PATH: str) -> List[str]:
    """
    Collect all .py and .ipynb files from a file or directory path.
    Args:
        INPUT_PATH (str): Path to a file or directory.
    Returns:
        List[str]: List of file paths matching .py or .ipynb extensions.
    """
    if os.path.isfile(INPUT_PATH):
        if INPUT_PATH.endswith(('.py', '.ipynb')):
            return [INPUT_PATH]
        else:
            return []
    elif os.path.isdir(INPUT_PATH):
        files = []
        for root, _, filenames in os.walk(INPUT_PATH):
            for fname in filenames:
                if fname.endswith(('.py', '.ipynb')):
                    files.append(os.path.join(root, fname))
        return files
    else:
        logging.error(f"Input path not found: {INPUT_PATH}")
        return []

def update_json(JSON_PATH: str, new_metadata: List[Dict[str, Any]]):
    """
    Append new metadata entries to the specified JSON file. Allows duplicates during development.
    Args:
        JSON_PATH (str): Path to the JSON metadata file.
        new_metadata (List[Dict[str, Any]]): List of metadata dictionaries to append.
    """
    try:
        if os.path.exists(JSON_PATH):
            with open(JSON_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        else:
            data = []
        # Development mode: allow duplicate entries for the same file
        for meta in new_metadata:
            if meta:
                data.append(meta)
        with open(JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logging.info(f"Updated {JSON_PATH} with {len(new_metadata)} new entries.")
    except Exception as e:
        logging.error(f"Error updating JSON file: {e}")

# --- Main CLI ---
def main():
    client = get_anthropic_client()
    files = collect_files(INPUT_PATH)
    if not files:
        logging.warning("No valid .py or .ipynb files found.")
        return
    results = []
    for file_path in files:
        meta = process_file(file_path, client)
        if meta:
            results.append(meta)
    if results:
        update_json(JSON_PATH, results)
    else:
        logging.info("No metadata extracted.")

if __name__ == '__main__':
    main()