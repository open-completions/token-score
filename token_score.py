from typing import Any, Dict

from pydantic import BaseModel


class TokenScoreMetrics(BaseModel):
    """Holds tokenization evaluation metrics of a tokenization pipeline,
    generally over a single programming language."""

    # The average number of bytes per token over the entire dataset.
    compression: float

    # A measure of the tokenizer's ability to split code identifiers at
    # canonical boundaries.
    #
    # Example: "numberOfUsers" -> ["number", "Of", "Users"]
    identifier_splitting_score: float

    # A measure of the tokenizer's ability to split the source code sequence
    # at canonical boundaries according to the language's grammar.
    #
    # Example: "if None: return" -> ["if", " ", "None", ":", " ", "return"]
    syntax_splitting_score: float

    # The total number of tokens in the dataset.
    total_tokens: int

    # The total number of bytes in the dataset.
    total_bytes: int


class TokenScore(BaseModel):
    """A structure that holds the result of the evaluation of a tokenization
    pipeline."""

    # The metrics for each programming language.
    metrics: Dict[str, TokenScoreMetrics]

    # Parity assesses the fairness of a tokenizer in treating equivalent
    # sequences across different programming languages.
    #
    # TODO: Add parity metrics.
    parity: Any

    # The total number of tokens in the dataset.
    total_tokens: int

    # The total number of bytes in the dataset.
    total_bytes: int
