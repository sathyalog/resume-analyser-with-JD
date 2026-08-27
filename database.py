import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv(override=True)

index_name = "resume-analyser"

@st.cache_resource
def get_pinecone_index():
    """Initializes Pinecone once and caches the index object across Streamlit rerenders."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if not pc.has_index(index_name):
        print(f"Creating index {index_name}..")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Index {index_name} is created :)")
    else:
        print(f"Index {index_name} already exists")

    # Return the connected index object directly
    return pc.Index(index_name)
