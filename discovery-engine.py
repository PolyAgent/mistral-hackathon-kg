import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, Settings
from langchain import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
import openai

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate




class FireworkLLM(LLM):

    @property
    def _llm_type(self) -> str:
        return "Firework"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        client = openai.OpenAI(
            base_url = os.environ["OPENAI_API_BASE"],
            api_key=os.environ["OPENAI_API_KEY"],
        )
        response = client.chat.completions.create(
          model="accounts/fireworks/models/mixtral-8x7b-instruct",
          temperature=0,
          max_tokens=4096*4,
          messages=[{
            "role": "user",
            "content": prompt,
          }],
        )
        return response.choices[0].message.content

        

class DiscoveryEngine:
    def __init__(self):
        self.QDRANT_URL = os.environ["QDRANT_URL"]
        self.QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
        self.OPENAI_API_BASE = os.environ["OPENAI_API_BASE"]
        self.OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
        self.COLLECTION_NAME = os.environ["QDRANT_COLLECTION_NAME"]
        self.index, self.retriever = self.get_index_and_retriever()
        self.llm = FireworkLLM()


    def get_index_and_retriever(self):
        qdrant_client = QdrantClient(url=self.QDRANT_URL, api_key=self.QDRANT_API_KEY)
        embed_model = OpenAIEmbedding(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            api_base=os.environ["OPENAI_API_BASE"],
            api_key=os.environ["OPENAI_API_KEY"])
        Settings.embed_model = embed_model

        vector_store = QdrantVectorStore(client=qdrant_client, collection_name=self.COLLECTION_NAME)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
        retriever = index.as_retriever()
        return index, retriever
    
    def get_relevant_documents(self, question: str) -> List[str]:
        documents = self.retriever.retrieve(question)
        sorted_documents = sorted(documents, key=lambda node: node.score, reverse=True)
        relevant_documents = [node.text.replace("\n", " ") for node in sorted_documents]
        return relevant_documents