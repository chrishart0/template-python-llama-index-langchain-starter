# template_gen_ai_project/examples/simple-llama-index-neo4j-graph.py
# https://docs.llamaindex.ai/en/stable/examples/property_graph/graph_store/

# TODO List
# - Use local LLM to generate triples

# Useful Cypher Commands
# - Match (n) Return n : show all nodes & relationships
# - Match (n) Detach Delete n : delete all nodes and relationships
# Wipe graph store
# MATCH (n) DETACH DELETE n
# CALL apoc.schema.assert({},{},true) YIELD label, key RETURN *


from llama_index.core import (
    Settings as LlamaIndexSettings,
)
from llama_index.core.indices import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from template_gen_ai_project.settings import settings
from template_gen_ai_project.helpers.logger_helper import get_logger
from llama_index.llms.azure_openai import AzureOpenAI
from template_gen_ai_project.helpers.embeddings import (
    setup_embeddings,
    Library,
    ModelProvider,
)

from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor


# from llama_index.readers.wikipedia import WikipediaReader

# Get the configured logger
logger = get_logger()

# https://docs.arize.com/phoenix/tracing/integrations-tracing/llamaindex
logger.info("Registering tracer provider")
tracer_provider = register(
    project_name="simple-llama-index-property-graph",
    endpoint="http://localhost:6006/v1/traces",
)
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

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
    library=Library.LlamaIndex,
    model_provider=ModelProvider.AzureOpenAI,
    # library=Library.LlamaIndex,
    # model_provider=ModelProvider.HuggingFace,
)
LlamaIndexSettings.chunk_size = 512

# global
username = "neo4j"
password = "pleaseletmein"
url = "bolt://localhost:7687"
database = "neo4j"
embed_dim = 1536
test_data_dir = "./test_data/graph_rag_docs/llama_index_overview"
index_cache_dir = ".cache/knowledge_graph_index"


logger.info(f"Connecting to Neo4j at {url}")
graph_store = Neo4jPropertyGraphStore(
    username=username,
    password=password,
    url=url,
    database=database,
)

# NOTE: can take a while!
logger.info("Connecting to index")

index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
)
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


# Interactive chat interface
while True:
    user_query = input("User: ")
    if user_query.lower() == "quit":
        # Disconnect from Neo4j
        graph_store.close()
        break
    query_graph_store(user_query)
