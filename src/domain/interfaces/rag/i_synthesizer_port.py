from abc import ABC, abstractmethod
from src.domain.value_objects.project_match_result import ProjectMatchResult
from src.domain.value_objects.sales_enablement_result import SalesEnablementResult
from src.domain.value_objects.profile_match_result import ProfileMatchResult
from src.domain.value_objects.technical_prep_result import TechnicalPrepResult


class ISynthesizerPort(ABC):
    """
    Port (Interface) for the LLM-based synthesis and reranking step.
    Implemented by GeminiSynthesizerAdapter in the infrastructure layer.

    Receives the full job description and chunk evidence from the
    retrieval pipeline, and returns ranked project match results.
    """

    @abstractmethod
    async def synthesize(
        self,
        job_details: str,
        chunk_evidence: list[dict],
    ) -> list[ProjectMatchResult]:
        """
        Use an LLM to evaluate chunk evidence against a job description
        and produce ranked project match results.

        Args:
            job_details: Plain text job description string.
            chunk_evidence: List of chunk dicts (with project metadata)
                            retrieved from Stage 2.

        Returns:
            List of ProjectMatchResult value objects, sorted by
            match_score descending.

        Raises:
            SynthesisError: If the LLM call fails or returns invalid data.
        """
        ...

    @abstractmethod
    async def generate_sales_enablement(
        self,
        job_details: str,
        projects: list[dict],
    ) -> SalesEnablementResult:
        """
        Use an LLM to generate sales enablement content given a job description
        and the list of matched projects.

        Args:
            job_details: Plain text job description string.
            projects: List of project dicts containing name, domain, tech_stack,
                      and description. Project names are used as internal context
                      only — they must NOT appear in the output.

        Returns:
            A SalesEnablementResult containing discovery_questions,
            talking_points, and outreach_template.

        Raises:
            SynthesisError: If the LLM call fails or returns invalid data.
        """
        ...

    @abstractmethod
    async def synthesize_profile_matches(
        self,
        job_details: str,
        variant_payloads: list[dict],
    ) -> list[ProfileMatchResult]:
        """
        Use an LLM to evaluate candidate variant payloads against a job
        description and produce ranked profile match results.

        Args:
            job_details: Plain text job description string.
            variant_payloads: List of full variant payload dicts retrieved
                              from Qdrant (containing candidate info, tech
                              stacks, projects, certifications, etc.).

        Returns:
            List of ProfileMatchResult value objects, sorted by
            match_percentage descending.

        Raises:
            SynthesisError: If the LLM call fails or returns invalid data.
        """
        ...

    @abstractmethod
    async def generate_technical_prep(
        self,
        job_details: str,
        candidate_context: dict,
    ) -> TechnicalPrepResult:
        """
        Use an LLM to generate a targeted Technical Preparation guide for a
        selected candidate ahead of a client interview.

        Args:
            job_details: Plain text job description string.
            candidate_context: Dict containing the candidate's full profile
                (name, title, experience, projects, tech_stacks) merged with
                the skill gap data from Step 3 (matching_skills, missing_skills).

        Returns:
            A TechnicalPrepResult with a technical_briefing_note and an
            ordered interview_preparation_guide (gaps first, strengths last).

        Raises:
            SynthesisError: If the LLM call fails or returns invalid data.
        """
        ...
