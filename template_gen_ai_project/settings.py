from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import Any, Optional
import os  # Import the os module
from template_gen_ai_project.helpers.logger_helper import get_logger

# Get the configured logger
logger = get_logger()

load_dotenv()


class Settings(BaseSettings):
    OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_DEPLOYMENT_NAME: Optional[str] = None
    AZURE_OPENAI_API_VERSION: Optional[str] = "2023-07-01-preview"
    OUTPUT_DIRECTORY: str = "./outputs/"  # New setting for output directory
    MODEL: str = "gpt-4o-mini"

    model_config = {"env_file": ".env"}

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not self.OPENAI_API_KEY and not (
            self.AZURE_OPENAI_API_KEY
            and self.AZURE_OPENAI_ENDPOINT
            and self.AZURE_OPENAI_DEPLOYMENT_NAME
        ):
            raise ValueError(
                "Either OPENAI_API_KEY is required or all 3 Azure parameters are required"
            )

        # Create the output directory if it doesn't exist
        if not os.path.exists(self.OUTPUT_DIRECTORY):
            logger.info(
                f"Creating output directory since it doesn't exist: {self.OUTPUT_DIRECTORY}"
            )
            os.makedirs(self.OUTPUT_DIRECTORY)


settings = Settings()
