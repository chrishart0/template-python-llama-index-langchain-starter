# template_gen_ai_project/examples/graph-examples/llama-index-knowledge-graph-query-engine.py
# https://docs.llamaindex.ai/en/stable/examples/query_engine/knowledge_graph_query_engine/

from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from template_gen_ai_project.settings import settings
from template_gen_ai_project.helpers.logger_helper import get_logger
from llama_index.core import Settings as LlamaIndexSettings
from llama_index.llms.azure_openai import AzureOpenAI
from template_gen_ai_project.helpers.embeddings import (
    setup_embeddings,
    Library,
    ModelProvider,
)

import logging
import sys

# Get the configured logger
logger = get_logger()

logging.basicConfig(
    stream=sys.stdout, level=logging.DEBUG
)  # logging.DEBUG for more verbose output


llm = AzureOpenAI(
    model=settings.AZURE_OPENAI_MODEL,
    temperature=0,
    deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)

LlamaIndexSettings.llm = llm
LlamaIndexSettings.embed_model = setup_embeddings(
    library=Library.LlamaIndex, model_provider=ModelProvider.AzureOpenAI
)
LlamaIndexSettings.chunk_size = 512

# global
username = "neo4j"
password = "pleaseletmein"
url = "bolt://localhost:7687"
database = "neo4j"
embed_dim = 1536
test_data_dir = "./test_data/financial_docs"
index_cache_dir = "./knowledge_graph_index"

logger.info("Connecting to Neo4j at {url}")
graph_store = Neo4jGraphStore(
    username=username,
    password=password,
    url=url,
    database=database,
)

logger.info("Creating storage context")
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# query_engine = KnowledgeGraphQueryEngine(
#     storage_context=storage_context,
#     llm=llm,
#     verbose=True,
# )

graph_rag_retriever = KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    verbose=True,
    graph_traversal_depth=1,
)

query_engine = RetrieverQueryEngine.from_args(
    graph_rag_retriever,
    verbose=True,
)

logger.info("Querying knowledge graph")


def query_knowledge_graph(query):
    logger.info(f"User: {query}")
    response = query_engine.query(
        query,
    )
    print("\n")
    print(f"Agent: {response}")
    print("\n")


# query_knowledge_graph("Tell me about Peter Quill?")


# Interactive chat interface
while True:
    user_query = input("User: ")
    if user_query.lower() == "quit":
        break
    query_knowledge_graph(user_query)
