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
from VCPilot import VCPilot
import random
from typing import List

# Set up environment variables
os.environ["QDRANT_URL"] = st.secrets.qdrant.url
os.environ["QDRANT_API_KEY"] = st.secrets.qdrant.api_key
os.environ["OPENAI_API_BASE"] = st.secrets.fireworks.base_url
os.environ["OPENAI_API_KEY"] = st.secrets.fireworks.api_key
os.environ["ANTHROPIC_API_KEY"] = st.secrets.anthropic.api_key

st.title("Discovery Engine")
agent = VCPilot()

st.markdown("""You can:
- Type the area of reasearch you are interested in.
""")

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import pymongo
from pymongo import MongoClient
client = MongoClient(st.secrets.mongodb.uri)
client.list_database_names()
db = client["arxiv"]
papers = db["papers_for_review"]

def get_triplets (citations):
  all_triplets = []
#   print(f"citations to process: {citations}")
  for citation in citations:
    # print(f"citation: {citation}")
    cited_text = citation.text
    # print(f"cited text: {cited_text}")
    #print(cited_text);
    #json_content = json.loads(node_content)
    title = cited_text.split("\n", 1)[0][:-1]
    # print(f"found triplets for {title}")
    cited_paper = papers.find_one({"title":title})
    print(f"cited paper: {cited_paper}")
    if(cited_paper and cited_paper.get('triples')):
      all_triplets.append(cited_paper['triples'])
      print(cited_paper['triples'])

  return all_triplets

def generate_graph(proposal, citations, summaries, llm):
    #json.loads(citations)
    #json_triplets = json.dumps(citations)
    graph_prompt_string = f"""
Based on these triplets, build a knowledge graph, ground it in the topic of interest: {proposal}
{citations}
Use ASCII symbols to visualize interconnections (nodes, edges, relationships with arrows >). Visualize it so it looks like a visual graph.


"""



    graph_prompt = "\n".join([graph_prompt_string])
    print(graph_prompt)
    response = llm.invoke(graph_prompt)
    return response

if proposal := st.chat_input("Insert keywords or abstract"):
    st.chat_message("user").markdown(proposal)
   
    try:
        with st.spinner("Generating research tasks..."):
            tasks = agent.get_research_tasks(proposal)
            # st.chat_message("assistant").markdown(tasks)
        with st.spinner("Agent pull relevant citations and triplets"):
            # citations = agent.get_relevant_documents(proposal)
            agent_executor = agent.get_agent_executor()
            summaries, citations = agent.get_research(proposal, agent_executor, tasks)
            # print(f"summaries: {summaries}")
            # print(f"citations: {citations}")
            final_triplets = get_triplets(citations)
            graph = generate_graph(proposal, final_triplets, summaries, agent.llm)
            st.chat_message("assistant").markdown(graph)
            
            # citations = agent.get_relevant_documents(proposal)
            # st.chat_message("assistant").markdown(citations)
            # final_triplets = get_triplets(citations)
            # st.chat_message("assistant").markdown(final_triplets)
        # with st.spinner("Initializing agent..."):
        #     agent_executor = agent.get_agent_executor()
        # with st.spinner("Agent performes research..."):
        #     summaries, citations2 = get_research(proposal, agent_executor, tasks)
        # with st.spinner("Agent generates graph..."):
        #     graph = agent.generate_graph(proposal, final_triplets, summaries)
        #     st.chat_message("assistant").markdown(graph)
        # with st.spinner("Agent performing research..."):
        #     summaries, citations = vcpilot.get_research(question, agent_executor, tasks)
        # with st.spinner("Getting highlights from research..."):
        #     highlights = vcpilot.generate_highlights(question, citations, summaries)
        # with st.spinner("Considering areas for followup..."):
        #     followups = vcpilot.get_followup_questions(highlights)
        # with st.spinner("Wrapping up..."):
        #     conclusion = vcpilot.get_conclusion(question, highlights)
        # tasks_str = "- " + "\n- ".join(tasks)
        # with st.spinner("Initializing agent..."):
        #     time.sleep(2)
        #     agent_executor = discoveryengine.get_agent_executor()
        # st.chat_message("assistant").markdown("some cool stuff comes here")
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
