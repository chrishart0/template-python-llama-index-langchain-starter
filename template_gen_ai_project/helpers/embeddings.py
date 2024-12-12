# template_gen_ai_project/helpers/embeddings.py

from template_gen_ai_project.settings import settings
from template_gen_ai_project.helpers.logger_helper import get_logger
from enum import Enum
from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding as LlamaIndexHuggingFaceEmbedding,
)
from llama_index.embeddings.azure_openai import (
    AzureOpenAIEmbedding as LlamaIndexAzureOpenAIEmbedding,
)

# Get the configured logger
logger = get_logger()


class Library(Enum):
    LlamaIndex = "llama_index"
    Langchain = "langchain"


class ModelProvider(Enum):
    AzureOpenAI = "azure_openai"
    HuggingFace = "huggingface"


class HuggingFaceEmbeddingModel(Enum):
    Jina = "jina"
    BAAI = "baai"


class LlamaIndexEmbeddingModels:
    def AzureOpenAIEmbedding():
        return LlamaIndexAzureOpenAIEmbedding(
            model=settings.AZURE_OPENAI_EMBEDDING_MODEL,
            deployment_name=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )

    def HuggingFaceEmbedding(
        model: HuggingFaceEmbeddingModel = HuggingFaceEmbeddingModel.Jina,
    ):
        # Embedding Model Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
        if model == HuggingFaceEmbeddingModel.Jina:
            logger.info("Using Jina Embeddings")
            logger.warning(
                "Jina Embeddings runs with trust_remote_code=True, we aren't sure if this is safe."
            )
            # TODO: Document why we need trust_remote_code=True
            # https://huggingface.co/jinaai/jina-embeddings-v3/discussions/22
            return LlamaIndexHuggingFaceEmbedding(
                # https://jina.ai/news/jina-embeddings-v3-a-frontier-multilingual-embedding-model/
                #   NOTE: You may need to delete your cache for the roberta flash implementation:
                #   `rm -rf /Users/$USER/.cache/huggingface/modules/transformers_modules/jinaai/xlm-roberta-flash-implementation`
                model_name="jinaai/jina-embeddings-v3",
                trust_remote_code=True,
            )
        elif model == HuggingFaceEmbeddingModel.BAAI:
            logger.info("Using BAAI Embeddings")
            return LlamaIndexHuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        else:
            raise ValueError(f"Invalid HuggingFaceEmbeddingModel: {model}")


def setup_embeddings(library: Library, model_provider: ModelProvider):
    """
    Set up the embeddings model based on the provided parameters.

    :param library: The library to use for setting up the embeddings model.
                    Options are: Library.LlamaIndex.
    :param model_provider: The model provider to use for setting up the embeddings model.
                           Options are: "azure_openai", "huggingface".
    :return: The configured embeddings model.
    """
    if library == Library.LlamaIndex:
        if model_provider == ModelProvider.AzureOpenAI:
            logger.info("Using Azure OpenAI Embeddings")
            return LlamaIndexEmbeddingModels.AzureOpenAIEmbedding()

        elif model_provider == ModelProvider.HuggingFace:
            logger.info("Using Hugging Face Embeddings")
            return LlamaIndexEmbeddingModels.HuggingFaceEmbedding()
    elif library == Library.Langchain:
        logger.error(
            "Langchain embeddings are not implemented yet, if you see this please implement them."
        )
        raise NotImplementedError(
            "Langchain embeddings are not implemented yet, if you see this please implement them."
        )
    else:
        logger.error(
            f"The library {library} is not supported, if you see this please implement it."
        )
        raise NotImplementedError(
            f"The library {library} is not supported, if you see this please implement it."
        )
