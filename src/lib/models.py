import uuid
from typing import List
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A single source of information: a character span inside a file."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without an answer."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions (answered or not)."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Sources retrieved for a given question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results plus the generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Search results for a set of questions, retrieved with top-k = k."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results with answers for a set of questions."""

    search_results: List[MinimalAnswer]
    k: int
 