import functools
import signal
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from pydantic import BaseModel
from spiral import ronin
from tiktoken import Encoding as OAIEncoding
from transformers import BatchEncoding as HFEncoding
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from tree_sitter import Language as TSLanguage
from tree_sitter import Node as TSNode
from tree_sitter import Parser as TSParser
from tree_sitter import Tree as TSTree
from tree_sitter_languages import get_language as ts_get_language

HFTokenizer = PreTrainedTokenizerFast | PreTrainedTokenizer

# The set of languages supported by TokenScore.
SUPPORTED_LANGUAGES = set(
    [
        "c++",
        "go",
        "java",
        "javascript",
        "python",
    ]
)


class Token(BaseModel):
    """A token is a byte range over a source code document"""

    range: Tuple[int, int]


class SyntaxToken(Token):
    """A structure that holds a syntax token and its byte range over the
    document's content."""

    # The token's syntax type.
    type: str


class Document(BaseModel):
    """A structure that holds the content of a source code file."""

    # The language of the document.
    lang: str

    # The content of the document.
    content: bytes

    def parse(self) -> TSTree:
        """Returns the AST of the document."""
        return TS_PARSERS[self.lang].parse(self.content)

    def token_to_bytes(self, token: Token) -> bytes:
        """Returns the bytes of the token."""
        return self.content[token.range[0] : token.range[1]]

    def token_to_string(self, token: Token) -> str:
        """Returns the string of the token."""
        return self.token_to_bytes(token).decode("utf-8")


class TokenScoreMetrics(BaseModel):
    """Holds tokenization evaluation metrics of a tokenization pipeline,
    generally over a single programming language."""

    # The average number of bytes per token over the entire dataset.
    compression: float

    # The average ratio of tokens per source code identifier.
    identifier_fertility: float

    # A measure of the tokenizer's ability to split code identifiers at
    # canonical boundaries.
    #
    # Example: "numberOfUsers" -> ["number", "Of", "Users"]
    identifier_splitting_score: float

    # A measure of the tokenizer's ability to split code identifiers at
    # canonical boundaries but using the raw tokens produced by the tokenizer.
    raw_identifier_splitting_score: float

    # A measure of the tokenizer's ability to split the source code sequence
    # at canonical boundaries according to the language's grammar. It is
    # computed by looking at the average number of syntax tokens that a single
    # token spans.
    #
    # Example: "if None: return" -> ["if", " ", "None", ":", " ", "return"]
    token_span_score: float

    # The total number of tokens in the file.
    total_tokens: int

    # The total number of bytes in the file.
    total_bytes: int


class IdentifierSplits(BaseModel):
    identifier: SyntaxToken

    raw_tokenizer_splits: List[str]

    tokenizer_splits: List[str]

    authoritative_splits: List[str]


@dataclass
class TokenScore:
    """All the artefacts produced when computing token score."""

    tree: TSTree

    identifiers: List[SyntaxToken]

    identifier_splits: List[IdentifierSplits]

    syntax_tokens: List[SyntaxToken]

    metrics: TokenScoreMetrics


def timeout(seconds=5):
    """A decorator that raises a TimeoutError if the decorated function takes
    longer than `seconds` to run."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def handle_timeout(signum, frame):
                raise TimeoutError()

            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(seconds)
            result = func(*args, **kwargs)
            signal.alarm(0)
            return result

        return wrapper

    return decorator


@timeout(10)
def compute_token_score(
    document: Document, tokens: List[Token], return_token_span_score: bool = True
) -> TokenScore:
    """Computes the token score of document."""

    tree = document.parse()

    identifiers = collect_identifiers(tree, document)

    syntax_tokens = []
    if return_token_span_score:
        syntax_tokens = collect_syntax_tokens(tree, document.content)

    compression = 0
    if len(tokens) != 0:
        compression = len(document.content) / len(tokens)

    (
        identifier_splitting_score,
        raw_identifier_splitting_score,
        identifier_fertility,
        identifier_splits,
    ) = compute_identifier_splitting_score(document, identifiers, tokens)

    token_span_score = 0
    if return_token_span_score:
        token_span_score = compute_token_span_score(syntax_tokens, tokens)

    return TokenScore(
        metrics=TokenScoreMetrics(
            compression=compression,
            identifier_fertility=identifier_fertility,
            identifier_splitting_score=identifier_splitting_score,
            raw_identifier_splitting_score=raw_identifier_splitting_score,
            token_span_score=token_span_score,
            total_tokens=len(tokens),
            total_bytes=len(document.content),
        ),
        tree=tree,
        syntax_tokens=syntax_tokens,
        identifier_splits=identifier_splits,
        identifiers=identifiers,
    )


def tokens_overlap(a: Token, b: Token) -> bool:
    """Returns true if the two tokens overlap."""
    return a.range[0] < b.range[1] and b.range[0] < a.range[1]


def tiktoken_tokenizer(enc: OAIEncoding, document: Document) -> List[Token]:
    ids = enc.encode_ordinary(document.content.decode("utf-8", errors="strict"))
    tokens = []

    offset = 0
    for i in range(len(ids)):
        b = enc.decode_single_token_bytes(ids[i])
        tokens.append(Token(range=(offset, offset + len(b))))
        offset += len(b)

    return tokens


def huggingface_tokenizer(tokenizer: HFTokenizer, document: Document) -> List[Token]:
    decoded_document = document.content.decode("utf-8", errors="strict")

    enc: HFEncoding = tokenizer.encode_plus(
        decoded_document,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation="do_not_truncate",
    )

    byte_offset_mapping = []
    last_char_offset = None

    assert (
        len(enc.offset_mapping) == len(enc.input_ids)
    ), f"len offset mapping {len(enc.offset_mapping)} != len input_ids {len(enc.input_ids)}"

    for char_start, char_end in enc.offset_mapping:
        # Decode only the new part of the text to find the byte length
        if last_char_offset != (char_start, char_end):
            char_start = last_char_offset[1] if last_char_offset else 0
            char_byte_length = len(
                decoded_document[char_start:char_end].encode("utf-8")
            )
            last_char_offset = (char_start, char_end)
        else:
            char_byte_length = 0

        if byte_offset_mapping:
            byte_start = byte_offset_mapping[-1][1]
        else:
            byte_start = 0

        byte_end = byte_start + char_byte_length

        byte_offset_mapping.append((byte_start, byte_end))

    # Convert byte_offset_mapping to a list of Token instances
    tokens = [Token(range=offset) for offset in byte_offset_mapping]

    assert len(tokens) == len(
        enc.input_ids
    ), f"len tokens {len(tokens)} != len input_ids {len(enc.input_ids)}"
    assert tokens[-1].range[1] == len(
        document.content
    ), f"last token {tokens[-1].range[1]} end != len document {len(document.content)}"

    return tokens


def compute_token_span_score(
    syntax_tokens: List[SyntaxToken], tokens: List[Token]
) -> float:
    """Computes the token span score of a document."""

    token_span_score_sum = 0

    for token in tokens:
        for syntax_token in syntax_tokens:
            if tokens_overlap(token, syntax_token):
                token_span_score_sum += 1
            if token.range[1] < syntax_token.range[0]:
                break

    token_span_score = 0
    if len(tokens) != 0:
        token_span_score = token_span_score_sum / len(tokens)

    return token_span_score


def compute_identifier_splitting_score(
    document: Document,
    identifiers: List[SyntaxToken],
    tokens: List[Token],
) -> Tuple[float, float, float, List[IdentifierSplits]]:
    """Computes the identifier splitting score of a document."""

    jaccard_similarity_count = 0
    jaccard_similarity_sum = 0

    raw_jaccard_similarity_count = 0
    raw_jaccard_similarity_sum = 0

    identifier_fertility_count = 0
    identifier_fertility_sum = 0

    identifier_splits = []

    for identifier in identifiers:
        try:
            # This shouldn't happen as code identifiers are generally valid
            # UTF-8.
            identifier_str = document.token_to_string(identifier)
        except UnicodeDecodeError:
            continue

        raw_tokenizer_splits = [
            document.token_to_bytes(token).decode("utf-8", errors="ignore")
            for token in tokens
            if identifier.range[0] < token.range[1]
            and token.range[1] <= identifier.range[1]
        ]

        identifier_fertility_count += 1
        identifier_fertility_sum += len(raw_tokenizer_splits)

        # Find all the tokens that span the identifier's byte range.
        tokenizer_splits = [
            # We create a new token to ensure that, when considering the
            # identifier "abc" in the snippet "let abc = 10;", the token
            # "abc" is not polluted by any extra characters that would come
            # after or before it.
            # The rationale for ignoring errors is that if a token is not valid
            # UTF-8 then it's by definition not a correct split.
            document.token_to_bytes(
                Token(
                    range=(
                        max(token.range[0], identifier.range[0]),
                        min(token.range[1], identifier.range[1]),
                    )
                )
            )
            .decode("utf-8", errors="ignore")
            .replace("_", "")
            for token in tokens
            if tokens_overlap(token, identifier)
        ]

        tokenizer_splits = list(filter(None, tokenizer_splits))

        authoritative_splits = ronin.split(identifier_str)

        jaccard_similarity_count += 1
        jaccard_similarity_sum += compute_jaccard_similarity_score(
            set(tokenizer_splits), set(authoritative_splits)
        )

        raw_jaccard_similarity_count += 1
        raw_jaccard_similarity_sum += compute_jaccard_similarity_score(
            set(raw_tokenizer_splits), set(authoritative_splits)
        )

        identifier_splits.append(
            IdentifierSplits(
                identifier=identifier,
                tokenizer_splits=tokenizer_splits,
                raw_tokenizer_splits=raw_tokenizer_splits,
                authoritative_splits=authoritative_splits,
            )
        )

    jaccard = 0
    if jaccard_similarity_count != 0:
        jaccard = jaccard_similarity_sum / jaccard_similarity_count

    raw_jaccard = 0
    if raw_jaccard_similarity_count != 0:
        raw_jaccard = raw_jaccard_similarity_sum / raw_jaccard_similarity_count

    identifier_fertility = 0
    if identifier_fertility_count != 0:
        identifier_fertility = identifier_fertility_sum / identifier_fertility_count

    return jaccard, raw_jaccard, identifier_fertility, identifier_splits


def collect_identifiers(tree: TSTree, document: Document) -> List[SyntaxToken]:
    """Collects the identifiers of the AST and their byte ranges over the
    document's content."""

    TS_LANGUAGES[document.lang].node_kind_for_id

    query = TS_LANGUAGES[document.lang].query(__TS_QUERIES[document.lang])

    matches = query.captures(tree.root_node)

    identifiers = [
        SyntaxToken(range=(match[0].start_byte, match[0].end_byte), type=match[0].type)
        for match in matches
    ]

    return identifiers


def collect_syntax_tokens(tree: TSTree, content: bytes) -> List[SyntaxToken]:
    """Collects the leaf nodes of the AST and their byte ranges over the
    document's content."""
    syntax_tokens = []

    prev_start_byte = 0
    prev_end_byte = 0

    def collect_tokens(node: TSNode):
        nonlocal prev_start_byte, prev_end_byte

        if node.child_count == 0:
            token_type = str(node.type)
            token_range = (node.start_byte, node.end_byte)

            if prev_end_byte != node.start_byte:
                syntax_tokens.append(
                    SyntaxToken(range=(prev_end_byte, node.start_byte), type="unknown")
                )

            syntax_tokens.append(SyntaxToken(range=token_range, type=token_type))

            prev_start_byte = node.start_byte
            prev_end_byte = node.end_byte
        else:
            for child in node.children:
                collect_tokens(child)

    collect_tokens(tree.root_node)

    if prev_end_byte < len(content):
        syntax_tokens.append(
            SyntaxToken(range=(prev_end_byte, len(content)), type="unknown")
        )

    return syntax_tokens


def compute_jaccard_similarity_score(set1: Set[str], set2: Set[str]) -> float:
    """Calculate the Jaccard Similarity between two sets of splits."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    if union == 0:
        return 0

    return intersection / union


def __build_languages() -> Dict[str, TSLanguage]:
    """Builds a mapping from language name to tree-sitter language."""
    languages = {}
    for lang in SUPPORTED_LANGUAGES:
        languages[lang] = ts_get_language(__TREE_SITTER_LANGUAGE_SLUGS[lang])
    return languages


def __build_parsers() -> Dict[str, TSParser]:
    """Builds a TreeSitter parser for each language in the given list."""

    parsers = {}
    for lang in SUPPORTED_LANGUAGES:
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

TS_LANGUAGES = __build_languages()

TS_PARSERS = __build_parsers()
