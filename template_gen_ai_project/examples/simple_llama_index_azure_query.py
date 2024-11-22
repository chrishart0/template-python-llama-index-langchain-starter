from llama_index.llms.azure_openai import AzureOpenAI

from llama_index.core import Settings as LlamaIndexSettings
from template_gen_ai_project.settings import settings

LlamaIndexSettings.llm = AzureOpenAI(
    engine=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
    model=settings.MODEL,
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)
