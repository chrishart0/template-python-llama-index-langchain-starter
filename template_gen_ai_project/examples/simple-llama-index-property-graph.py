# template_gen_ai_project/examples/simple-llama-index-neo4j-graph.py
# https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/Neo4jKGIndexDemo/

# TODO List
# - Use a modern embeddings model
# - Use local LLM to generate triples
# - Implement tracing to understand what's happening under the hood

# Useful Cypher Commands
# - Match (n) Return n : show all nodes & relationships
# - Match (n) Detach Delete n : delete all nodes and relationships

from llama_index.core import (
    StorageContext,
    SimpleDirectoryReader,
)
from llama_index.core.indices import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from template_gen_ai_project.settings import settings
from template_gen_ai_project.helpers.logger_helper import get_logger
from llama_index.core import Settings as LlamaIndexSettings
from llama_index.llms.azure_openai import AzureOpenAI
from template_gen_ai_project.helpers.embeddings import (
    setup_embeddings,
    Library,
    ModelProvider,
)

# from llama_index.readers.wikipedia import WikipediaReader

# Get the configured logger
logger = get_logger()

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
    #        library=Library.LlamaIndex, model_provider=ModelProvider.AzureOpenAI
    library=Library.LlamaIndex,
    model_provider=ModelProvider.HuggingFace,
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

# Prepare documents
# loader = WikipediaReader()
#
# logger.info("Loading Wikipedia data")
# documents = loader.load_data(
#    pages=["Guardians of the Galaxy Vol. 3"],
#    auto_suggest=False,
# )
#
documents = SimpleDirectoryReader("./test_data/llamaindex").load_data()

logger.info(f"Connecting to Neo4j at {url}")
graph_store = Neo4jPropertyGraphStore(
    username=username,
    password=password,
    url=url,
    database=database,
)

logger.info("Creating storage context")
storage_context = StorageContext.from_defaults(graph_store=graph_store)


# NOTE: can take a while!
logger.info("Creating index")
index = PropertyGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    # max_triplets_per_chunk=2, # Makes ingestion faster, but not as good
    show_progress=True,
    property_graph_store=graph_store,
)
storage_context.persist(persist_dir="./.cache/knowledge_graph_index")

# logger.info(f"Loading index from {index_cache_dir}")
# index = load_index_from_storage(
#     StorageContext.from_defaults(persist_dir=index_cache_dir)
# )


###########################


query_engine = index.as_query_engine(
    # include_text=False,
    # response_mode="tree_summarize",
    # embedding_mode="hybrid",
    # similarity_top_k=5,
    verbose=True,
)

logger.info("Querying with LLM")


def query_graph_store(query):
    print("\n")
    logger.info(f"User: {query}")
    print(f"Agent: {query_engine.query(query)}")
    print("\n")


# Example queries
# query_graph_store("How was the quarter?")
# query_graph_store("What is the document about?")

# Interactive chat interface
while True:
    user_query = input("User: ")
    if user_query.lower() == "quit":
        break
    query_graph_store(user_query)
