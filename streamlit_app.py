import streamlit as st
from docx import Document
from pinecone import Pinecone, ServerlessSpec
import openai
import time
from openai import OpenAI
import tiktoken
from tiktoken import get_encoding
import os
from dotenv import load_dotenv
import random

# Load environment variables from .env file
load_dotenv()

# Access your API key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

# List of example questions
EXAMPLE_QUESTIONS = [
    "How to optimize shop layout",
    "How to clean Graco sprayer",
    "How to avoid mistakes with LR32 system",
    "Best practices for wood finishing",
    "How to choose the right wood for a project",
    "Tips for precise miter cuts",
    "Effective dust collection methods",
    "How to sharpen woodworking tools",
    "Techniques for joining wood without screws",
    "Best wood glues for different applications"
]

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to extract metadata from text
def extract_metadata_from_text(text):
    title = text.split('\n')[0] if text else "Untitled Video"  # Use the first line as title
    return {
        "title": title,
    }

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "bents-woodworking"

# Check if the index already exists before creating it
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(INDEX_NAME)

# Function to truncate text
def truncate_text(text, max_tokens):
    tokenizer = get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return tokenizer.decode(tokens[:max_tokens])

# Function to count tokens
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function to chunk text
def chunk_text(text, max_tokens):
    tokenizer = get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk))
    return chunks

# Function to generate embeddings with retries
def generate_embedding(text):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            truncated_text = truncate_text(text, 8000)  # Truncate to 8000 tokens
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=truncated_text
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Error creating embedding after {max_retries} attempts: {str(e)}")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff

# Function to upsert data into Pinecone
def upsert_transcript(transcript_text, metadata):
    chunks = chunk_text(transcript_text, 8000)
    for i, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk)
        if embedding:
            chunk_metadata = metadata.copy()
            chunk_metadata['text'] = chunk
            chunk_metadata['chunk_id'] = f"{metadata['title']}_chunk_{i}"
            index.upsert([(chunk_metadata['chunk_id'], embedding, chunk_metadata)])

# Function to query Pinecone
def query_pinecone(query, index):
    query_embedding = generate_embedding(query)
    if query_embedding:
        result = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
        return result['matches']
    else:
        return []

# Function to get answer from GPT-4
def get_answer(context, user_query):
    max_context_tokens = 3000
    truncated_context = truncate_text(context, max_context_tokens)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """You are an assistant expert representing Jason Bent on woodworking based on
            information uploaded in the document. You are an AI assistant focused on explaining answers to questions based on
            how Jason Bent would answer. At any time you provide a response, include citations for title of the video the
            information is from and the timestamp of where the relevant information is presented from. Provide response as if
            you are Jason Bent in that particular tense. Also make sure to post affiliate links as included below in your response
            only if the response you are building has any of these names:
            TSO Products - https://tsoproducts.com/?aff=5
            Bits and Bits Company - 10% off with code BENTS10 - http://bit.ly/bitsbitsbw
            Taylor Toolworks - https://lddy.no/1e5hv
            Festool LR 32 System - https://amzn.to/3hRTvLB
            Festool Trigger Clamp - https://amzn.to/2HoVydC
            Festool LR 32 Rail - https://amzn.to/33LsnsG
            Festool OF 1400 - https://amzn.to/2FRerp5
            Festool Vac Sys Head - https://amzn.to/3010rjw
            Festool Midi Vac - https://amzn.to/2HfMmrM
            Festool Bluetooth Switch - https://amzn.to/33RAt36
            Woodpeckers TS600 - https://amzn.to/3mIc34t"""},
            {"role": "user", "content": f"Answer the following question based on the context: {truncated_context}\n\nQuestion: {user_query}"}
        ]
    )
    return response.choices[0].message.content.strip()

# Function to process the query and display the answer
def process_query(query):
    if query:
        with st.spinner("Searching for the best answer..."):
            matches = query_pinecone(query, index)
            if matches:
                retrieved_texts = [match['metadata']['text'] for match in matches]
                retrieved_titles = [match['metadata']['title'] for match in matches]
                context = " ".join([f"Title: {title}\n{text}" for title, text in zip(retrieved_titles, retrieved_texts)])
                final_answer = get_answer(context, query)
                st.subheader("Jason's Answer:")
                st.write(final_answer)
                # Extract sources from the answer
                mentioned_sources = set()
                for title in retrieved_titles:
                    if title in final_answer:
                        mentioned_sources.add(title)
                # Update chat history
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                st.session_state.chat_history.append((query, final_answer))
            else:
                st.warning("I couldn't find a specific answer to your question. Please try rephrasing or ask something else.")
    else:
        st.warning("Please enter a question before searching.")

# Streamlit Interface
st.set_page_config(page_title="Bent's Woodworking Assistant", layout="wide")

# Add the logo to the main page
st.image("bents logo.png", width=150)
st.title("Bent's Woodworking Assistant")

# Sidebar for file upload and metadata
with st.sidebar:
    st.header("Upload Transcripts")
    uploaded_files = st.file_uploader("Upload YouTube Video Transcripts (DOCX)", type="docx", accept_multiple_files=True)
    if uploaded_files:
        all_metadata = []
        total_token_count = 0
        for uploaded_file in uploaded_files:
            transcript_text = extract_text_from_docx(uploaded_file)
            metadata = extract_metadata_from_text(transcript_text)
            all_metadata.append((metadata, transcript_text))
            token_count = num_tokens_from_string(transcript_text)
            total_token_count += token_count
        st.subheader("Uploaded Transcripts")
        for metadata, _ in all_metadata:
            st.text(f"Title: {metadata['title']}")
        st.text(f"Total token count: {total_token_count}")
        if st.button("Upsert All Transcripts"):
            with st.spinner("Upserting transcripts..."):
                for metadata, transcript_text in all_metadata:
                    upsert_transcript(transcript_text, metadata)
            st.success("All transcripts upserted successfully!")

# Initialize selected questions in session state
if 'selected_questions' not in st.session_state:
    st.session_state.selected_questions = random.sample(EXAMPLE_QUESTIONS, 3)

# Display popular questions
st.header("Popular Questions")
for question in st.session_state.selected_questions:
    if st.button(question, key=question):
        process_query(question)

st.header("Custom Question")
user_query = st.text_input("What would you like to know about woodworking?")
if st.button("Get Answer"):
    process_query(user_query)

# Add a section for displaying recent questions and answers
if 'chat_history' in st.session_state and st.session_state.chat_history:
    st.header("Recent Questions and Answers")
    for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-5:])):
        with st.expander(f"Q: {q}"):
            st.write(f"A: {a}")
