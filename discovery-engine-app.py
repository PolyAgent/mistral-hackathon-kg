import streamlit as st
import requests
import openai

import logging
import sys
import os
import time
import qdrant_client
from IPython.display import Markdown, display
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document, Settings
import os
from qdrant_client import QdrantClient
import openai

# Set up environment variables
os.environ["QDRANT_URL"] = st.secrets.qdrant.url
os.environ["QDRANT_API_KEY"] = st.secrets.qdrant.api_key
os.environ["OPENAI_API_BASE"] = st.secrets.fireworks.base_url
os.environ["OPENAI_API_KEY"] = st.secrets.fireworks.api_key
os.environ["ANTHROPIC_API_KEY"] = st.secrets.anthropic.api_key

st.title("Discovery Engine")

st.markdown("""You can:
- Type the keywords or full abstract of a paper
""")

if question := st.chat_input("Insert keywords or abstract"):
    st.chat_message("user").markdown(question)
   
    try:
        with st.spinner("detecting entities..."):
            time.sleep(2)
            # extracting NERs
        with st.spinner("Generating knowledge graph..."):
            time.sleep(2)
        # with st.spinner("Initializing agent..."):
        #     time.sleep(2)
        #     agent_executor = discoveryengine.get_agent_executor()
        final_report = f"""
some cool stuff comes here
"""
        st.chat_message("assistant").markdown(final_report)
    except Exception as e:
        responses = [
           "Oops! Something went wrong while attempting to build the knowledge graph. Please try again later.",
            "Uh-oh! It seems we've hit a snag while building the knowledge graph. Please refresh the page and try again.",
            "Sorry, we encountered an unexpected issue while processing your request to build the knowledge graph. Please try another time.",
            "Houston, we have a problem! We couldn't build the knowledge graph due to an unexpected error. Please stand by while we investigate.",
            "Yikes! It looks like there was a hiccup while building the knowledge graph. We're working to resolve the issue. Please try again shortly.",
            ]
        st.chat_message("assistant").markdown(random.choice(responses) + f" error: {e}")
