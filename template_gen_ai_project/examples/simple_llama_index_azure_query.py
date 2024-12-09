import os

from llama_index.llms.azure_openai import AzureOpenAI

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core import Settings as LlamaIndexSettings
from llama_index.core import get_response_synthesizer
from llama_index.core.node_parser.text.token import TokenTextSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from template_gen_ai_project.settings import settings
from template_gen_ai_project.helpers.logger_helper import get_logger

logger = get_logger()
PERSIST_DIR = "./cache"

LlamaIndexSettings.llm = AzureOpenAI(
    engine=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
    model=settings.MODEL,
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)

LlamaIndexSettings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

if not os.path.exists(PERSIST_DIR):
    logger.info("Cache doesn't exist. Creating cache")

    logger.info("Ingesting Documents")
    docs = SimpleDirectoryReader(input_files=["./data/10k/lyft_2021.pdf"]).load_data()
    parser = TokenTextSplitter()
    pipeline = IngestionPipeline(
        transformations=[
            parser,
            HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        ]
    )
    nodes = pipeline.run(documents=docs, show_progress=True)

    logger.info("Building index")
    lyft_index = VectorStoreIndex(nodes)

    lyft_index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    logger.info("Cache detected. Loading existing index")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    lyft_index = load_index_from_storage(storage_context)

logger.info("Retrieving Nodes")
retriever = VectorIndexRetriever(index=lyft_index, similarity_top_k=3)

query = "What information does this document contain?"

nodes = retriever.retrieve(query)

print(type(nodes[0]))

for node in nodes:
    print(f"Node ID: {node.node_id}, Text: {node.text}")

response_synthesizer = get_response_synthesizer(response_mode="compact")

response = response_synthesizer.synthesize(query, nodes)

print(response)

# query_engine_tools = [
#    QueryEngineTool(
#        query_engine=lyft_engine,
#        metadata=ToolMetadata(
#            name="lyft_10k",
#            description=(
#                "Provides information about Lyft financials for year 2021"
#                "Provides information about Lyft financials"
#            )
#        )
#    ),
#    QueryEngineTool(
#        query_engine=uber_engine,
#        metadata=ToolMetadata(
#            name="uber_10k",
#            description=(
#                "Provides information about Uber financials for year 2021"
#            )
#        )
#    )
# ]
#
# s_engine = SubQuestionQueryEngine.from_defaults(
#    query_engine_tools=query_engine_tools,
#    verbose=True
# )
#
# logger.info("Running queries")
# response = s_engine.query(
#    "What information does this document contain?"
# )
# print(response)
