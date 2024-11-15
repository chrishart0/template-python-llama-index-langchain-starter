# template_gen_ai_project/examples/llama_index_structured_outputs_llm_response.py
# Source: https://docs.llamaindex.ai/en/stable/examples/structured_outputs/structured_outputs/
# Written with llama_index = "0.11.23"
# Call with: python3

from pydantic import BaseModel, Field
from typing import List
from template_gen_ai_project.settings import settings
from template_gen_ai_project.helpers.logger_helper import get_logger
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader

# Get the configured logger
logger = get_logger()

# Configure LLM
llm = OpenAI(
    model="gpt-4o-mini",
    api_key=settings.OPENAI_API_KEY,
)


reader = SimpleDirectoryReader(input_dir="test_data/medical_data/")
docs = reader.load_data()


# skip chunking since we're doing page-level chunking
index = VectorStoreIndex(docs)

# from llama_index.postprocessor.flag_embedding_reranker import (
#     FlagEmbeddingReranker,
# )

# reranker = FlagEmbeddingReranker(
#     top_n=5,
#     model="BAAI/bge-reranker-large",
# )


class Output(BaseModel):
    """Output containing the response, page numbers, and confidence."""

    response: str = Field(..., description="The answer to the question.")
    source_page_numbers: List[int] = Field(
        ...,
        description="The page numbers of the sources used to answer this question. Do not include a page number if the context is irrelevant.",
    )
    confidence: float = Field(
        ...,
        description="Confidence value between 0-1 of the correctness of the result.",
    )
    confidence_explanation: str = Field(
        ..., description="Explanation for the confidence score"
    )


sllm = llm.as_structured_llm(output_cls=Output)


query_engine = index.as_query_engine(
    similarity_top_k=5,
    # node_postprocessors=[reranker],
    llm=sllm,
    response_mode="tree_summarize",  # you can also select other modes like `compact`, `refine`
)

response = query_engine.query("What is the patient's name?")
print(response.response.dict())
