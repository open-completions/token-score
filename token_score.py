from typing import Any, Dict, List, Set, Tuple

from pydantic import BaseModel
from tree_sitter import Language as TSLanguage
from tree_sitter import Node as TSNode
from tree_sitter import Parser as TSParser
from tree_sitter import Tree as TSTree
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

    def parse(self) -> TSTree:
        """Returns the AST of the document."""
        return TS_PARSERS[self.lang].parse(self.content)


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


def collect_identifiers(
    tree: TSTree, document: Document
) -> List[DocumentSemanticToken]:
    """Collects the identifiers of the AST and their byte ranges over the
    document's content."""

    TS_LANGUAGES[document.lang].node_kind_for_id

    query = TS_LANGUAGES[document.lang].query(__TS_QUERIES[document.lang])

    matches = query.captures(tree.root_node)

    identifiers = [
        DocumentSemanticToken(
            range=(match[0].start_byte, match[0].end_byte), type=match[0].type
        )
        for match in matches
    ]

    return identifiers


def collect_semantic_tokens(
    tree: TSTree, content: bytes
) -> List[DocumentSemanticToken]:
    """Collects the leaf nodes of the AST and their byte ranges over the
    document's content."""
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

    return semantic_tokens


def compute_jaccard_similarity_score(set1: Set[str], set2: Set[str]) -> float:
    """Calculate the Jaccard Similarity between two sets of splits."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


def __build_languages() -> Dict[str, TSLanguage]:
    """Builds a mapping from language name to tree-sitter language."""
    languages = {}
    for lang in LANGUAGES:
        languages[lang] = ts_get_language(__TREE_SITTER_LANGUAGE_SLUGS[lang])
    return languages


def __build_parsers() -> Dict[str, TSParser]:
    """Builds a TreeSitter parser for each language in the given list."""

    parsers = {}
    for lang in LANGUAGES:
        try:
            parsers[lang] = TSParser()
            parsers[lang].set_language(TS_LANGUAGES[lang])
        except Exception as e:
            print(f"Error building parser for {lang}: {e}")

    return parsers


__TREE_SITTER_LANGUAGE_SLUGS = {
    "c++": "cpp",
    "go": "go",
    "java": "java",
    "javascript": "javascript",
    "python": "python",
}

TS_LANGUAGES = __build_languages()

TS_PARSERS = __build_parsers()

__TS_QUERIES = {
    "python": """
        (identifier) @id
    """,
    "go": """
        (identifier) @id
        (package_identifier) @pkg_id
        (type_identifier) @type_id
        (field_identifier) @field_id
        """,
    "java": """
        (identifier) @id
        (type_identifier) @type_id
        """,
    "javascript": """
        (identifier) @id
        (property_identifier) @property_id
    """,
    "c++": """
        (identifier) @id
        (type_identifier) @type_id
        (namespace_identifier) @namespace_id
        (field_identifier) @field_id
    """,
}
