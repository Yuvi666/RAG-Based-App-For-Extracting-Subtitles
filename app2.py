import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai
from langchain_core.output_parsers import StrOutputParser

# Streamlit app title
st.title("AI-Powered Query App 🤖🧠")
st.subheader("Search Anything From Your Database...🔍")

# Initialize database
model_name="sentence-transformers/all-mpnet-base-v2"
embedding_model=HuggingFaceEmbeddings(model_name=model_name)

db = Chroma(collection_name="vector_database", embedding_function=embedding_model, persist_directory="./chroma_db_")

# Input field for user query
query = st.text_input("Enter your query:")

# Process query on button click
if st.button("Search"):
    if not query:
        st.error("Please enter a query.")
    else:
        try:
            # Search the database
            docs_chroma = db.similarity_search_with_score(query, k=3)
            context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

            # Display context
            st.write("### Context Information:")
            st.text_area("Context", context_text, height=200)

            st.success("Query processed successfully!")
        except Exception as e:
            st.error(f"Error processing query: {e}")
