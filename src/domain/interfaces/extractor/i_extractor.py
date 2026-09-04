from abc import ABC, abstractmethod
from src.domain.value_objects.job_details import JobDetails


class IExtractor(ABC):
    """
    Port (Interface) for AI-based job data extraction.
    Implemented by GeminiExtractor in the infrastructure layer.
    """

    @abstractmethod
    async def extract(self, cleaned_text: str) -> JobDetails:
        """
        Send the cleaned page text to an LLM and return a structured JobDetails object.

        Args:
            cleaned_text: The plain text of the job page after HTML cleaning.

        Returns:
            A JobDetails value object with all extracted fields.

        Raises:
            ExtractionFailedException: If the LLM fails to produce valid structured output.
        """
        ...
