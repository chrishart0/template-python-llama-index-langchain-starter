# template_gen_ai_project/examples/medical_test_data_extraction.py
# Source: https://python.langchain.com/docs/how_to/structured_output/#choosing-between-multiple-schemas
# Written with langchain = "0.3.7"

from template_gen_ai_project.settings import settings
from template_gen_ai_project.helpers.logger_helper import get_logger
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import json
import tiktoken
from langchain_community.document_loaders import PDFPlumberLoader


# Get the configured logger
logger = get_logger()

# Configure LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=settings.OPENAI_API_KEY,
)


def estimate_openai_cost(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_million: float = 3.30,
    output_cost_per_million: float = 13.20,
) -> float:
    """
    Estimate the cost of using OpenAI's API based on token count.

    :param input_tokens: The number of input tokens used.
    :param output_tokens: The number of output tokens used.
    :param input_cost_per_million: The cost per million input tokens.
    :param output_cost_per_million: The cost per million output tokens.
    :return: The estimated cost.
    """
    input_cost = (input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (output_tokens / 1_000_000) * output_cost_per_million
    return input_cost + output_cost


class MedicalTest(BaseModel):
    """Information about a single medical test."""

    test_name: str = Field(description="The name of the medical test")
    test_type: Optional[str] = Field(
        description="The type of medical test, e.g., 'blood test', 'imaging', 'biopsy', etc."
    )
    date_conducted: Optional[datetime] = Field(
        description="The date when the medical test was conducted"
    )
    result: Optional[str] = Field(description="The result of the medical test")
    normal_range: Optional[str] = Field(
        description="The normal range for the test result"
    )
    unit: Optional[str] = Field(
        description="The unit of measurement for the test result"
    )
    notes: Optional[str] = Field(
        description="Any additional notes or observations about the test"
    )


class PatientReport(BaseModel):
    """A report containing multiple medical tests for a single patient."""

    patient_name: Optional[str] = Field(description="The name of the patient")
    doctor: Optional[str] = Field(
        description="The doctor who ordered or conducted the tests"
    )
    facility: Optional[str] = Field(
        description="The medical facility where the tests were conducted"
    )
    date_of_report: Optional[datetime] = Field(
        description="The date when the report was generated"
    )
    tests: List[MedicalTest] = Field(
        description="A list of medical tests included in the report"
    )
    report_notes: Optional[str] = Field(
        description="Any additional notes or observations about the report"
    )


# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)

runnable = prompt | llm.with_structured_output(schema=PatientReport)

# Inject the medical report test_data/medical_data/2024-nov-bloodtest-results-health-360.pdf
logger.info("Loading the medical report...")
loader = PDFPlumberLoader(
    "./test_data/medical_data/2024-nov-bloodtest-results-health-360.pdf"
)
docs = loader.load()

# Assemble the report text from the document
report_text = "\n".join([doc.page_content for doc in docs])
logger.info(report_text)

# Count tokens using tiktoken
# Use a known encoding, such as 'cl100k_base' for models like gpt-3.5-turbo
encoding = tiktoken.get_encoding("cl100k_base")
tokens = encoding.encode(report_text)
input_token_count = len(tokens)
logger.info(f"Input token count: {input_token_count}")

logger.info("Invoking the data extraction...")
response = runnable.invoke({"text": report_text})

# Calculate output tokens
response_text = response.json()  # Assuming response is a JSON object
output_tokens = encoding.encode(json.dumps(response_text))
output_token_count = len(output_tokens)
logger.info(f"Output token count: {output_token_count}")

# Estimate and print the OpenAI cost
estimated_cost = estimate_openai_cost(input_token_count, output_token_count)
logger.info(f"Estimated OpenAI cost: ${estimated_cost:.4f}")

logger.info("Writing the response to a JSON file...")
# Convert the Pydantic object to a dictionary before writing to JSON
response_dict = response.dict()
with open(
    "./test_data/medical_data/2024-nov-bloodtest-results-health-360.json", "w"
) as f:
    json.dump(response_dict, f, indent=4, default=str)
