# template_gen_ai_project/examples/graph-examples/llama-index-knowledge-graph-query-engine.py
# https://docs.llamaindex.ai/en/stable/examples/query_engine/knowledge_graph_query_engine/

from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import (
    KnowledgeGraphIndex,
    StorageContext,
)
from template_gen_ai_project.settings import settings
from template_gen_ai_project.helpers.logger_helper import get_logger
from llama_index.core import Settings as LlamaIndexSettings
from llama_index.llms.azure_openai import AzureOpenAI
from template_gen_ai_project.helpers.embeddings import (
    setup_embeddings,
    Library,
    ModelProvider,
)
from llama_index.readers.wikipedia import WikipediaReader

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

logger.info(f"Connecting to Neo4j at {url}")
graph_store = Neo4jGraphStore(
    username=username,
    password=password,
    url=url,
    database=database,
)

logger.info("Creating storage context")
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# Prepare documents
# documents = SimpleDirectoryReader(
#     "./test_data/financial_docs"
# ).load_data()

loader = WikipediaReader()

logger.info("Loading Wikipedia data")
documents = loader.load_data(
    pages=["Guardians of the Galaxy Vol. 3"],
    auto_suggest=False,
    verbose=True,
)

logger.info("Creating KnowledgeGraphIndex")
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=10,
    include_embeddings=True,
    show_progress=True,
    verbose=True,
)
