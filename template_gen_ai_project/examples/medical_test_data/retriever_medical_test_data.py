# template_gen_ai_project/examples/medical_test_data/retriever_medical_test_data.py
# Original Source: https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/
from template_gen_ai_project.settings import settings
from template_gen_ai_project.helpers.logger_helper import get_logger

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core import Settings as LlamaIndexSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOpenAI

import os

# Get the configured logger
logger = get_logger()

# ToDo: Move this to llm helper
llm = AzureOpenAI(
    engine=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
    model=settings.MODEL,
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)
# global
LlamaIndexSettings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en")
LlamaIndexSettings.llm = llm

# load data
medical_test_data_dir = f"{settings.OUTPUT_DIRECTORY}medical_test_data/"
medical_test_data_files = os.listdir(medical_test_data_dir)

query_engine_tools = []

for file in medical_test_data_files:
    medical_test_data = SimpleDirectoryReader(
        input_dir=os.path.join(medical_test_data_dir, file)
    ).load_data()

    # build index and query engine
    logger.info(f"Building index and query engine for {file}")
    vector_query_engine = VectorStoreIndex.from_documents(
        medical_test_data,
        llm=llm,
        use_async=True,
    ).as_query_engine()

    query_engine_tools.append(
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name=file,
                description=f"Medical Tests Data for a patient from {file}",
            ),
        ),
    )

logger.info(f"Query engine tools: {query_engine_tools}")

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=True,
)

response = query_engine.query(
    "What is the name of the patient and what their blood pressure is?"
)

logger.info(f"Response: {response}")
