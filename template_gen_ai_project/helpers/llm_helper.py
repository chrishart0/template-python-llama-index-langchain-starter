from template_gen_ai_project.settings import settings
from template_gen_ai_project.helpers.logger_helper import get_logger
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_ollama.chat_models import ChatOllama
from pydantic import BaseModel
from enum import Enum

# Get the configured logger
logger = get_logger()


class LLMType(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


class LLMSetupParams(BaseModel):
    llm_type: LLMType  # The type of LLM to set up. Can be 'openai' or 'ollama'.
    model: str  # The name of the model to use.
    use_langchain: bool  # Whether to use Langchain or Llama Index.


def setup_llm(params: LLMSetupParams):
    """
    Set up the Language Model (LLM) based on the provided parameters.

    :param params: The parameters to use for setting up the LLM.
    :return: The configured LLM.
    """
    if params.llm_type == "openai":
        if settings.OPENAI_API_KEY:
            logger.info("Using OpenAI API")
            llm = ChatOpenAI(
                model=params.model,
                api_key=settings.OPENAI_API_KEY,
            )
        elif (
            settings.AZURE_OPENAI_API_KEY
            and settings.AZURE_OPENAI_ENDPOINT
            and settings.AZURE_OPENAI_DEPLOYMENT_NAME
        ):
            logger.info("Using Azure OpenAI API")
            llm = AzureChatOpenAI(
                model=params.model,
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key=settings.AZURE_OPENAI_API_KEY,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                openai_api_version="2024-02-01",
            )
        else:
            raise ValueError(
                "Either OPENAI_API_KEY is required or all 3 Azure parameters are required"
            )
    elif params.llm_type == "ollama":
        logger.info("Using Ollama API")
        llm = ChatOllama(
            model=params.model,
            base_url="http://localhost:11434",
            temperature=0.1,
            timeout=100000,
            num_ctx=10000,
            verbose=True,
        )
    else:
        raise ValueError("Invalid LLM type. Choose 'openai' or 'ollama'.")

    return llm
