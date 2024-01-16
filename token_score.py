from typing import Any, Dict, List, Set, Tuple

from pydantic import BaseModel
from tree_sitter import Node as TSNode
from tree_sitter import Parser as TSParser
from tree_sitter_languages import get_language as ts_get_language

# The set of languages supported by TokenScore.
LANGUAGES = set(
    [
        "c++",
        "go",
        "java",
        "javascript",
        "python",
    ]
)


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


class Document(BaseModel):
    """A structure that holds the content of a source code file."""

    # The language of the document.
    lang: str

    # The content of the document.
    content: bytes


class DocumentToken(BaseModel):
    """A structure that holds a token and its byte range over the document's
    content."""

    # The token's byte range over the document's content.
    range: Tuple[int, int]

    def to_bytes(self, content: bytes) -> bytes:
        """Returns the bytes content of the token using the document's content."""
        return content[self.range[0] : self.range[1]]


class DocumentTokenizerResult(BaseModel):
    """The result of tokenizing a document using a tokenizer. Contains a list
    of bytes ranges over the document's content that correspond to each
    token."""

    # The list of tokens and their byte ranges over the document's content.
    tokens: List[DocumentToken]


class DocumentSemanticToken(BaseModel):
    """A structure that holds a semantic token and its byte range over the
    document's content."""

    # The token's byte range over the document's content.
    range: Tuple[int, int]

    # The token's semantic type.
    type: str

    def to_bytes(self, content: bytes) -> bytes:
        """Returns the bytes content of the token using the document's content."""
        return content[self.range[0] : self.range[1]]


class DocumentParserResult(BaseModel):
    """The result of parsing a document using a parser. Contains a list of
    byte ranges over the document's content that correspond to semantic AST
    tokens (e.g. identifiers, literals, operators, etc.)."""

    # The list of semantic tokens and their byte ranges over the document's
    # content.
    semantic_tokens: List[DocumentSemanticToken]


def compute_token_score(documents: Dict[str, Document]) -> TokenScore:
    """Computes the token score of a set of documents."""
    return TokenScore(
        metrics={},
        parity=None,
        total_tokens=0,
        total_bytes=0,
    )


def compute_document_parser_result(
    content: bytes, parser: TSParser
) -> DocumentParserResult:
    """Computes the parser result of a document using a parser."""
    tree = parser.parse(content)
    semantic_tokens = []

    prev_start_byte = 0
    prev_end_byte = 0

    def collect_tokens(node: TSNode):
        nonlocal prev_start_byte, prev_end_byte

        if node.child_count == 0:
            token_type = str(node.type)
            token_range = (node.start_byte, node.end_byte)

            if prev_end_byte != node.start_byte:
                semantic_tokens.append(
                    DocumentSemanticToken(
                        range=(prev_end_byte, node.start_byte), type="whitespace"
                    )
                )

            semantic_tokens.append(
                DocumentSemanticToken(range=token_range, type=token_type)
            )

            prev_start_byte = node.start_byte
            prev_end_byte = node.end_byte
        else:
            for child in node.children:
                collect_tokens(child)

    collect_tokens(tree.root_node)

    if prev_end_byte < len(content):
        semantic_tokens.append(
            DocumentSemanticToken(
                range=(prev_end_byte, len(content)), type="whitespace"
            )
        )

    return DocumentParserResult(semantic_tokens=semantic_tokens)


def build_parsers(languages: Set[str]) -> Dict[str, TSParser]:
    """Builds a TreeSitter parser for each language in the given list."""

    parsers = {}
    for lang in languages:
        try:
            parsers[lang] = TSParser()
            parsers[lang].set_language(ts_get_language(__TREE_SITTER_LANGUAGES[lang]))
        except Exception as e:
            print(f"Error building parser for {lang}: {e}")

    return parsers


__TREE_SITTER_LANGUAGES = {
    "c++": "cpp",
    "go": "go",
    "java": "java",
    "javascript": "javascript",
    "python": "python",
}

PARSERS = build_parsers(LANGUAGES)
