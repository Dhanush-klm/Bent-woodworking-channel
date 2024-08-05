import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import openai
import time
from openai import OpenAI
import tiktoken
from tiktoken import get_encoding
import os
from dotenv import load_dotenv
import random
import pandas as pd
from docx import Document
from database_operations import get_database_connection, init_db, load_initial_data, get_all_products, query_db_for_keywords, add_product, delete_product, update_product, get_product_by_id

# Load environment variables and initialize clients
load_dotenv()
# Access your API key
OPENAI_API_KEY = st.secrets("OPENAI_API_KEY")
PINECONE_API_KEY = st.secrets("PINECONE_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "bents-woodworking"

# Initialize Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(name=INDEX_NAME, dimension=1536, metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-east-1'))
index = pc.Index(INDEX_NAME)

# YouTube video links
YOUTUBE_LINKS = {
    "Basics of Cabinet Building": "https://www.youtube.com/watch?v=Oeu7ogH2NZU&t=3910s",
    "Graco Ultimate Sprayer": "https://www.youtube.com/watch?v=T8BIpNzdh7M&t=264s",
    "Festool LR32 system": "https://www.youtube.com/watch?v=EO62T1LHdNA"
}

# List of example questions
EXAMPLE_QUESTIONS = [
    "How do TSO Products' Festool accessories improve woodworking precision?",
    "What makes Bits and Bits Company's router bits ideal for woodworking?",
    "What are the benefits of Japanese saws and chisels from Taylor Toolworks?",
    "How does the Festool LR 32 System aid in cabinet making?",
    "What advantages does the Festool Trigger Clamp offer for quick release and one-handed use?",
    "How does the Festool LR 32 Rail ensure precise 32mm hole spacing?",
    "What features of the Festool OF 1400 plunge router are ideal for precision routing?",
    "How does the Festool Vac Sys Head improve clamping in woodworking?",
    "What makes the Festool Midi Vac a top choice for dust extraction?",
    "How does the Festool Bluetooth Switch enhance dust extractor control?"
]

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to extract metadata from text
def extract_metadata_from_text(text):
    title = text.split('\n')[0] if text else "Untitled Video"
    return {"title": title}

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
            truncated_text = truncate_text(text, 8000)
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=truncated_text
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Error creating embedding after {max_retries} attempts: {str(e)}")
                return None
            time.sleep(2 ** attempt)

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
        return [(match['metadata']['title'], match['metadata']['text']) for match in result['matches']]
    else:
        return []

# Function to generate keywords
def generate_keywords(text):
    keyword_prompt = f"Generate 3-5 highly relevant and specific keywords or short phrases from this text, separated by commas. Focus on technical terms, tool names, or specific woodworking techniques: {text}"
    keyword_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a keyword extraction assistant specialized in woodworking terminology. Generate concise, highly relevant keywords or short phrases from the given text."},
            {"role": "user", "content": keyword_prompt}
        ]
    )
    keywords = keyword_response.choices[0].message.content.strip().split(',')
    return [keyword.strip().lower() for keyword in keywords if keyword.strip()]

# Function to get answer from GPT-4o
def get_answer(context, user_query):
    max_context_tokens = 3000
    truncated_context = truncate_text(context, max_context_tokens)
    
    query_keywords = generate_keywords(user_query)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant expert representing Jason Bent on woodworking based on information uploaded in the document. You are an AI assistant focused on explaining answers to questions based on how Jason Bent would answer. At any time you provide a response, include citations for title of the video the information is from and the timestamp of where the relevant information is presented from. Provide response as if you are Jason Bent in that particular tense. Do not include any links or URLs in your response."},
            {"role": "user", "content": f"Answer the following question based on the context: {truncated_context}\n\nQuestion: {user_query}"}
        ]
    )
    initial_answer = response.choices[0].message.content.strip()
    
    answer_keywords = generate_keywords(initial_answer)
    
    all_keywords = list(set(query_keywords + answer_keywords))
    
    related_products = query_db_for_keywords(all_keywords)
    
    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant expert representing Jason Bent on woodworking. Incorporate the related product information if relevant, but do not include any links or URLs in your response."},
            {"role": "user", "content": f"Initial Answer: {initial_answer}\n\nRelated Products: {related_products}\n\nPlease provide a final answer that incorporates information from the related products, if relevant, without mentioning specific product names or including any links."}
        ]
    )
   
    final_answer = final_response.choices[0].message.content.strip()
    
    return final_answer, related_products, all_keywords

def process_query(query):
    if query:
        with st.spinner("Searching for the best answer..."):
            matches = query_pinecone(query, index)
            if matches:
                retrieved_texts = [text for _, text in matches]
                retrieved_titles = [title for title, _ in matches]
                context = " ".join([f"Title: {title}\n{text}" for title, text in zip(retrieved_titles, retrieved_texts)])
                final_answer, related_products, keywords = get_answer(context, query)
                st.write(final_answer)
                
                # Display YouTube video as a smaller miniplayer
                for title in retrieved_titles:
                    if title in YOUTUBE_LINKS:
                        st.subheader(f"Related Video: {title}")
                        video_id = YOUTUBE_LINKS[title].split("v=")[1].split("&")[0]
                        st.markdown(f"""
                            <iframe width="320" height="180" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
                        """, unsafe_allow_html=True)
                        break  # Display only the first matching video
                
                st.subheader("Related Products:")
                if related_products:
                    for product in related_products:
                        st.markdown(f"[{product[1]}]({product[3]})")
                else:
                    st.write("No related products found.")
                
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                st.session_state.chat_history.append((query, final_answer, related_products))
            else:
                st.warning("I couldn't find a specific answer to your question. Please try rephrasing or ask something else.")
    else:
        st.warning("Please enter a question before searching.")

def query_interface():
    st.header("Woodworking Assistant")

    # Initialize selected questions and current query in session state
    if 'selected_questions' not in st.session_state:
        st.session_state.selected_questions = random.sample(EXAMPLE_QUESTIONS, 3)
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""

    # Display popular questions
    st.subheader("Popular Questions")
    for question in st.session_state.selected_questions:
        if st.button(question, key=question):
            st.session_state.current_query = question

    # User input and Get Answer button
    user_query = st.text_input("What would you like to know about woodworking?")
    if st.button("Get Answer"):
        st.session_state.current_query = user_query

    # Process the query and display response
    if st.session_state.current_query:
        st.subheader("Response:")
        process_query(st.session_state.current_query)
        # Clear the query after processing
        st.session_state.current_query = ""

    # Add a section for displaying recent questions and answers
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        st.header("Recent Questions and Answers")
        for i, (q, a, products) in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"Q: {q}"):
                st.write(f"A: {a}")
                if products:
                    st.subheader("Related Products:")
                    for product in products:
                        st.markdown(f"[{product[1]}]({product[3]})")

def database_interface():
    st.header("Database Management")

    # Display all products
    st.subheader("All Products")
    products = get_all_products()
    if products:
        df = pd.DataFrame(products, columns=['ID', 'Title', 'Tags', 'Links'])
        st.dataframe(df)
    else:
        st.write("The database is empty.")

    # Add new product
    st.subheader("Add New Product")
    new_title = st.text_input("Title")
    new_tags = st.text_input("Tags (comma-separated)")
    new_link = st.text_input("Link")
    if st.button("Add Product"):
        if new_title and new_tags and new_link:
            add_product(new_title, new_tags, new_link)
            st.success("Product added successfully!")
        else:
            st.warning("Please fill in all fields.")

    # Update product
    st.subheader("Update Product")
    update_id = st.number_input("Enter Product ID to update", min_value=1, step=1)
    product = get_product_by_id(update_id)
    if product:
        update_title = st.text_input("New Title", value=product[1])
        update_tags = st.text_input("New Tags", value=product[2])
        update_link = st.text_input("New Link", value=product[3])
        if st.button("Update Product"):
            update_product(update_id, update_title, update_tags, update_link)
            st.success(f"Product with ID {update_id} updated successfully!")
    else:
        st.warning(f"No product found with ID {update_id}")

    # Delete product
    st.subheader("Delete Product")
    delete_id = st.number_input("Enter Product ID to delete", min_value=1, step=1)
    if st.button("Delete Product"):
        delete_product(delete_id)
        st.success(f"Product with ID {delete_id} deleted successfully!")

# Main Streamlit app
def main():
    st.set_page_config(page_title="Bent's Woodworking Assistant", layout="wide")

    # Add the logo to the main page
    st.image("bents logo.png", width=150)
    st.title("Bent's Woodworking Assistant")

    # Initialize database connection
    conn = get_database_connection()
    init_db(conn)
    load_initial_data()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["Query Interface", "Database Management"])

    if page == "Query Interface":
        query_interface()
    elif page == "Database Management":
        database_interface()

    # Sidebar for file upload and metadata (only in Query Interface)
    if page == "Query Interface":
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

if __name__ == "__main__":
    main()
