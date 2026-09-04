"""Value objects for the Technical Preparation pipeline."""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


@dataclass
class InterviewTopic:
    """A single topic in the interview preparation guide.

    Attributes:
        topic:     The title of the preparation topic (e.g. "React Native Core").
        focus:     Always "weakness" (candidate needs to learn this).
        questions: 3-5 study/preparation questions for this skill, calibrated
                   to the years of experience demanded by the job description.
    """
    topic: str
    focus: str          # "weakness"
    questions: list[str] = field(default_factory=list)


@dataclass
class TechnicalPrepResult:
    """Output of the Technical Preparation generation pipeline.

    Attributes:
        technical_briefing_note:   A 2-3 sentence plain-English summary
                                   describing what the interview will focus on,
                                   referencing the candidate by name.
        interview_preparation_guide: Ordered list of InterviewTopic objects
                                     (weakness topics only).
    """
    technical_briefing_note: str
    interview_preparation_guide: list[InterviewTopic] = field(
        default_factory=list
    )


class InterviewTopicOutput(BaseModel):
    """Pydantic schema used to enforce Gemini's structured output."""

    topic: str
    focus: Literal["weakness"] = Field(
        description='Always "weakness"'
    )
    questions: list[str] = Field(
        description=(
            "3-5 study/preparation questions for this skill. Difficulty must be "
            "calibrated to the years of experience required in the job description."
        )
    )


class TechnicalPrepOutput(BaseModel):
    """Pydantic schema used to enforce Gemini's structured output."""

    technical_briefing_note: str
    interview_preparation_guide: list[InterviewTopicOutput]
