# This application is developed by Neo Mohsenvand, starting on Sep 3 2024. Claude Sonnet 3.5 was used to optimize the code. 

import streamlit as st
from streamlit_chat import message
import os
import time
from pathlib import Path
import chromadb
import json
import PyPDF2
import docx
import openpyxl
import chardet
from typing import List, Dict, Union
import io
from collections import Counter
import re
import openai

# Set page config at the very beginning
st.set_page_config(page_title="SimpleRAG", page_icon="💬", layout="wide")

# Constants for initiating the interface
EMBEDDING_MODELS = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
CHAT_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
DEFAULT_CHUNK_SIZE = 4000  # the number of characters in each chunk 
DEFAULT_CHUNK_OVERLAP = 2000  # the number of overlapping characters between two concecutive chunks
DEFAULT_NUM_CHUNKS = 4 # the number of chunks that are added to the chat context
DEFAULT_TEMPERATURE = 0.1 # the degree of model creativity (less = more strict)
DEFAULT_MEMORY_SIZE = 3 # the number of recent Q&A pairs to keep in memory and add to the chat context

# Default prompt template for the chat model. 
DEFAULT_PROMPT_TEMPLATE = """Here are the chunks retrieved based on similarity search from user's question. They might or might not be directly related to the question:
{context}

Document Summaries:
{summaries}

Chat History:
{memory}

User Question: {question}

Please provide a comprehensive answer to the user's question based on the given context, document summaries, and chat history. If the information is not available in the provided context, please state that you don't have enough information to answer the question.

Your Answer:"""

def embed_documents(api_key: str, texts: List[str], model: str) -> Union[List[List[float]], None]:
    """
    Generate embeddings for a list of text documents using OpenAI's API.

    Args:
        api_key (str): The OpenAI API key.
        texts (List[str]): A list of text documents to embed.
        model (str): The name of the embedding model to use.

    Returns:
        Union[List[List[float]], None]: A list of embeddings (each embedding is a list of floats),
                                        or None if there's an error.

    Note:
        This function uses the specified OpenAI embedding model.
        Errors are displayed in the Streamlit sidebar.
    """
    openai.api_key = api_key
    try:
        response = openai.Embedding.create(
            model=model,
            input=texts
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        st.sidebar.error(f"Error getting embedding: {str(e)}")
        return None

def chat_with_openai(api_key: str, model: str, prompt: str, temperature: float) -> str:
    """
    Generate a response from the OpenAI model based on the given prompt.

    Args:
        api_key (str): The OpenAI API key.
        model (str): The name of the chat model to use.
        prompt (str): The input prompt for the model.
        temperature (float): The temperature parameter for controlling randomness in the output.

    Returns:
        str: The generated response from the model.

    Note:
        This function uses OpenAI's ChatCompletion API.
        Errors are displayed in the Streamlit sidebar.
    """
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        st.sidebar.error(f"Error generating response: {str(e)}")
        return None

# File processing functions
def detect_encoding(file_content: bytes) -> str:
    """
    Detect the encoding of a given byte string.

    Args:
        file_content (bytes): The content of the file as bytes.

    Returns:
        str: The detected encoding of the file content.

    Note:
        This function uses the 'chardet' library to guess the encoding.
    """
    return chardet.detect(file_content)['encoding']

def read_file(file: io.BytesIO, filename: str) -> str:
    _, file_extension = os.path.splitext(filename)
    """
    Read and extract text content from various file types.

    Args:
        file (io.BytesIO): A file-like object containing the file data.
        filename (str): The name of the file, used to determine the file type.

    Returns:
        str: The extracted text content from the file.

    Note:
        Supports PDF, DOCX, XLSX, and plain text files.
        For unsupported file types, it attempts to read them as text files.
        Errors during file reading are displayed in the Streamlit sidebar.
    """
    try:
        if file_extension.lower() == '.pdf':
            pdf_reader = PyPDF2.PdfReader(file)
            return ' '.join([page.extract_text() for page in pdf_reader.pages])
        
        elif file_extension.lower() == '.docx':
            doc = docx.Document(file)
            return ' '.join([para.text for para in doc.paragraphs])
        
        elif file_extension.lower() in ['.xlsx', '.xls']:
            workbook = openpyxl.load_workbook(file)
            return ' '.join([str(cell.value) for sheet in workbook.worksheets for row in sheet.iter_rows() for cell in row if cell.value])
        
        else:  # Assume it's a text file
            content = file.read()
            encoding = detect_encoding(content)
            return content.decode(encoding)
    
    except Exception as e:
        st.sidebar.error(f"Error reading file {filename}: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split a long text into smaller overlapping chunks.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[str]: A list of text chunks.

    Note:
        This function is useful for processing long documents that exceed
        the token limit of language models.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

def summarize_document(api_key: str, model: str, text: str) -> str:
    """
    Generate a concise summary of a given document using the OpenAI model.

    Args:
        api_key (str): The OpenAI API key.
        model (str): The name of the model to use for summarization.
        text (str): The document text to summarize.

    Returns:
        str: A summary of the document, or an error message if summarization fails.

    Note:
        This function uses a fixed prompt template for summarization and
        a temperature of 0.1 for more deterministic outputs.
    """
    prompt = f"Please provide a concise summary of the following document:\n\n{text}\n\nSummary:"
    response = chat_with_openai(api_key, model, prompt, 0.1)
    if response:
        return response
    return "Failed to generate summary."

def process_documents(api_key: str, chat_model: str, files: List[io.BytesIO], chunk_size: int, chunk_overlap: int) -> List[Dict[str, Union[str, Dict[str, Union[str, int]]]]]:
    """
    Process a list of document files, extracting content, generating summaries, and chunking text.

    Args:
        api_key (str): The OpenAI API key for embeddings and summarization.
        chat_model (str): The name of the OpenAI model to use for summarization.
        files (List[io.BytesIO]): A list of file-like objects containing the documents.
        chunk_size (int): The maximum size of each text chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        Tuple[List[Dict], Dict]: A tuple containing:
            - A list of dictionaries, each representing a document chunk with its metadata.
            - A dictionary of document summaries, keyed by filename.

    Note:
        This function combines file reading, summarization, and text chunking operations.
    """
    documents = []
    summaries = {}
    for file in files:
        content = read_file(file, file.name)
        if content:
            summary = summarize_document(api_key, chat_model, content)
            summaries[file.name] = summary
            chunks = chunk_text(content, chunk_size, chunk_overlap)
            for i, chunk in enumerate(chunks):
                documents.append({
                    "content": chunk,
                    "metadata": {
                        "source": file.name,
                        "chunk": i,
                        "summary": summary
                    }
                })
    return documents, summaries

def get_chat_history(messages: List[Dict[str, str]], memory_size: int) -> str:
    """
    Format the recent chat history into a string representation.

    Args:
        messages (List[Dict[str, str]]): A list of message dictionaries, each containing 'role' and 'content'.
        memory_size (int): The number of recent message pairs to include in the history.

    Returns:
        str: A formatted string representation of the chat history.

    Note:
        This function is useful for providing context to the language model
        about recent interactions in the conversation.
    """
    history = messages[-memory_size*2:]  # Get last n question-answer pairs
    formatted_history = []
    for msg in history:
        role = "Human" if msg["role"] == "user" else "AI"
        formatted_history.append(f"{role}: {msg['content']}")
    return "\n".join(formatted_history)

def keyword_search(query: str, documents: List[Dict[str, Union[str, Dict[str, Union[str, int]]]]]) -> List[float]:
    """
    Perform a simple keyword-based search on a list of documents.

    Args:
        query (str): The search query.
        documents (List[Dict]): A list of document dictionaries, each containing a 'content' key.

    Returns:
        List[float]: A list of relevance scores for each document, based on keyword matching.

    Note:
        This function uses a basic intersection-over-union approach for scoring.
        It's a simple but effective method for keyword matching.
    """
    query_words = set(query.lower().split())
    scores = []
    for doc in documents:
        doc_words = set(doc['content'].lower().split())
        score = len(query_words.intersection(doc_words)) / len(query_words)
        scores.append(score)
    return scores

def hybrid_search(query: str, collection, documents: List[Dict[str, Union[str, Dict[str, Union[str, int]]]]], num_chunks: int):
    """
    Perform a hybrid search combining semantic similarity and keyword matching.

    Args:
        query (str): The search query.
        collection: A ChromaDB collection object for semantic search.
        documents (List[Dict]): A list of document dictionaries for keyword search.
        num_chunks (int): The number of top results to return.

    Returns:
        List[Tuple]: A list of tuples, each containing (document_content, metadata, combined_score).

    Note:
        This function combines the power of semantic search using embeddings
        with traditional keyword-based search for more robust results.
    """
    # Semantic search
    question_embedding = embed_documents(st.session_state.openai_api_key, [query], st.session_state.selected_embedding_model)
    semantic_results = collection.query(
        query_embeddings=question_embedding,
        n_results=num_chunks
    )

    # Keyword search
    keyword_scores = keyword_search(query, documents)

    # Combine results
    combined_results = []
    for i, (semantic_doc, keyword_score) in enumerate(zip(semantic_results['documents'][0], keyword_scores)):
        combined_score = 0.8*semantic_results['distances'][0][i] + 0.2*keyword_score  # weighted average of two scores
        combined_results.append((semantic_doc, semantic_results['metadatas'][0][i], combined_score))

    # Sort by combined score (lower is better)
    combined_results.sort(key=lambda x: x[2])

    return combined_results[:num_chunks]

def sanitize_collection_name(name: str) -> str:
    """
    Sanitize the collection name to meet ChromaDB's requirements.
    """
    # Remove any character that isn't alphanumeric, underscore, or hyphen
    sanitized = re.sub(r'[^\w-]', '_', name)
    # Ensure it starts and ends with an alphanumeric character
    sanitized = re.sub(r'^[^a-zA-Z0-9]+', '', sanitized)
    sanitized = re.sub(r'[^a-zA-Z0-9]+$', '', sanitized)
    # Limit to 63 characters
    sanitized = sanitized[:63]
    # Ensure it's at least 3 characters long
    while len(sanitized) < 3:
        sanitized += '_'
    return sanitized

def main():
    """
    The main function that sets up the Streamlit interface and handles the chat application flow.

    This function:
    1. Initializes session state variables
    2. Sets up the sidebar with various settings and file upload functionality
    3. Handles document processing and indexing
    4. Manages the chat interface, including displaying messages and handling user input
    5. Performs hybrid search on user queries and generates responses using the OpenAI model

    Note:
        This function serves as the entry point for the Streamlit application and
        orchestrates the entire SimpleRAG experience.
    """
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'summaries' not in st.session_state:
        st.session_state.summaries = {}

    with st.sidebar:
        st.title("💬 SimpleRAG")
        st.header("Settings")
        
        with st.expander("🤖 Model Settings", expanded=False):
            openai_api_key = st.text_input("OpenAI API Key:", type="password", key="openai_api_key_input")
            st.session_state.openai_api_key = openai_api_key

            # Add a selectbox for choosing the embedding model
            selected_embedding_model = st.selectbox(
                "Select OpenAI Embedding Model:",
                options=EMBEDDING_MODELS,
                index=0,
                key="embedding_model_selectbox"
            )
            st.session_state.selected_embedding_model = selected_embedding_model

            # Add a selectbox for choosing the chat model
            selected_chat_model = st.selectbox(
                "Select OpenAI Chat Model:",
                options=CHAT_MODELS,
                index=0,
                key="chat_model_selectbox"
            )
            st.session_state.selected_chat_model = selected_chat_model
            
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.1, key="temperature_slider")

        with st.expander("📄 Process Settings", expanded=False):
            chunk_size = st.number_input("Chunk size", min_value=100, max_value=4000, value=DEFAULT_CHUNK_SIZE, key="chunk_size_input")
            chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=2000, value=DEFAULT_CHUNK_OVERLAP, key="chunk_overlap_input")
            num_chunks = st.number_input("Number of chunks to retrieve", min_value=1, max_value=10, value=DEFAULT_NUM_CHUNKS, key="num_chunks_input")

        with st.expander("💬 Chat Settings", expanded=False):
            memory_size = st.number_input("Number of messages to keep in memory", min_value=1, max_value=10, value=DEFAULT_MEMORY_SIZE, key="memory_size_input")
            prompt_template = st.text_area("Prompt template", value=DEFAULT_PROMPT_TEMPLATE, key="prompt_template_input")

        uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, key="file_uploader")
        process_button = st.button("🔄 Process Documents", key="process_button")

        # Clear chat button in sidebar
        if st.button("🧹 Clear Chat", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

        if process_button and uploaded_files:
            if not openai_api_key:
                st.error("Please enter your OpenAI API key.")
            else:
                with st.spinner("Processing documents..."):
                    start_time = time.time()
                    
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Process uploaded documents
                    status_text.text("Reading and chunking documents...")
                    documents, summaries = process_documents(openai_api_key, st.session_state.selected_chat_model, uploaded_files, chunk_size, chunk_overlap)
                    progress_bar.progress(25)
                    
                    # Display processed files
                    st.subheader("Processed Files:")
                    for file in uploaded_files:
                        st.write(file.name)
                    
                    # Embed documents
                    status_text.text("Generating embeddings...")
                    embeddings = embed_documents(openai_api_key, [doc["content"] for doc in documents], st.session_state.selected_embedding_model)
                    progress_bar.progress(50)
                    
                    if embeddings:
                        # Initialize ChromaDB for vector storage
                        status_text.text("Initializing vector database...")
                        chroma_client = chromadb.Client()
                        progress_bar.progress(75)
                        
                        # Create a new collection with a sanitized name based on the embedding model
                        base_name = f"document_collection_{st.session_state.selected_embedding_model}"
                        collection_name = sanitize_collection_name(base_name)
                        
                        # Delete existing collection if it exists
                        try:
                            existing_collection = chroma_client.get_collection(name=collection_name)
                            if existing_collection:
                                chroma_client.delete_collection(name=collection_name)
                                st.warning(f"Deleted existing collection: {collection_name}")
                        except ValueError:
                            # Collection doesn't exist, so we can proceed
                            pass
                        
                        # Create a new collection
                        collection = chroma_client.create_collection(name=collection_name)
                        
                        # Add documents to ChromaDB
                        status_text.text("Indexing documents...")
                        collection.add(
                            embeddings=embeddings,
                            documents=[doc["content"] for doc in documents],
                            metadatas=[doc["metadata"] for doc in documents],
                            ids=[str(i) for i in range(len(documents))]
                        )
                        progress_bar.progress(100)
                        
                        end_time = time.time()
                        processing_time = end_time - start_time
                        status_text.text(f"Processing complete in {processing_time:.2f} seconds.")
                        
                        # Store processed data in session state
                        st.session_state.collection = collection
                        st.session_state.documents = documents
                        st.session_state.summaries = summaries
                        st.session_state.data_processed = True
                    else:
                        status_text.text("Failed to embed documents.")
                        st.error("Failed to embed documents.")

    # Chat interface
    st.header("Chat")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if st.session_state.data_processed:
        if prompt := st.chat_input("Ask a question about the documents"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if st.session_state.collection:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Hybrid search
                        results = hybrid_search(prompt, st.session_state.collection, st.session_state.documents, num_chunks)
                        
                        # Prepare context for the chat model
                        context = "\n\n".join([doc[0] for doc in results])
                        summaries_text = "\n\n".join([f"{file}: {summary}" for file, summary in st.session_state.summaries.items()])
                        chat_history = get_chat_history(st.session_state.messages, memory_size)
                        full_prompt = prompt_template.format(context=context, summaries=summaries_text, memory=chat_history, question=prompt)
                        
                        # Generate response using the selected OpenAI chat model
                        response = chat_with_openai(st.session_state.openai_api_key, st.session_state.selected_chat_model, full_prompt, temperature)
                        
                        if response:
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})

                            # Display used chunks in an expander
                            with st.expander("📚 View Relevant Document Chunks", expanded=False):
                                for i, (doc, metadata, score) in enumerate(results):
                                    st.markdown(f"**Chunk {i + 1}:**")
                                    st.text(doc)
                                    st.write(f"Source: {metadata['source']}")
                                    st.write(f"Chunk number: {metadata['chunk']}")
                                    st.write(f"Relevance score: {score:.4f}")
                                    st.markdown("---")
            else:
                st.error("Please process some documents first.")
    else:
        st.info("💡 Please enter your OpenAI API key and then upload and process documents to start chatting.")

if __name__ == "__main__":
    main()