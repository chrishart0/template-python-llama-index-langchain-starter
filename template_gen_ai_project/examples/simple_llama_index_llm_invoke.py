# tempalte_gen_ai_project/examples/llama_index_chat.py
# Source: https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_standalone/
# Written with llama_index = "0.11.23" 
# Call with: python3 

from template_gen_ai_project.settings import settings
from template_gen_ai_project.helpers.logger_helper import get_logger
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI


# Get the configured logger
logger = get_logger()

# Configure LLM
llm = OpenAI(
    model="gpt-4o-mini",
    api_key=settings.OPENAI_API_KEY,
)

##################################
#### Text Completion Examples ####
##################################

# non-streaming
completion = llm.complete("Paul Graham is ")
logger.info(completion)
logger.info("\n\n")

# using streaming endpoint

completions = llm.stream_complete("Paul Graham is ")
for completion in completions:
    logger.info(completion.delta, end="")

logger.info("\n\n")

#######################
#### Chat Examples ####
#######################

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.chat(messages)
logger.info(resp)
logger.info("\n\n")
