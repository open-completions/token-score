import argparse
from typing import List

import tiktoken
from rich.console import Console
from rich.table import Table

from token_score import Document, Token, compute_token_score

ENC = tiktoken.encoding_for_model("code")


def tokenizer(doc: Document) -> List[Token]:
    ids = ENC.encode_ordinary(doc.content.decode("utf-8", errors="strict"))
    token_bytes = [ENC.decode_single_token_bytes(id) for id in ids]
    tokens = []

    offset = 0
    for b in token_bytes:
        tokens.append(Token(range=(offset, offset + len(b))))
        offset += len(b)

    return tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    args = parser.parse_args()

    console = Console()

    doc = Document(lang=args.lang, content=open(args.file, "rb").read())

    tokens = tokenizer(doc)

    result = compute_token_score(doc, tokens)

    table = Table(show_header=True, header_style="bold magenta")

    table.add_column("Identifier")
    table.add_column("Expected Splits")
    table.add_column("Raw Tokenizer Splits")
    table.add_column("Result")
    table.add_column("Actual Splits")
    table.add_column("Result")

    for identifier_split in result.identifier_splits:
        table.add_row(
            doc.token_to_string(identifier_split.identifier),
            str(identifier_split.authoritative_splits),
            str(identifier_split.raw_tokenizer_splits),
            "✅"
            if identifier_split.authoritative_splits
            == identifier_split.raw_tokenizer_splits
            else "❌",
            str(identifier_split.tokenizer_splits),
            "✅"
            if identifier_split.authoritative_splits
            == identifier_split.tokenizer_splits
            else "❌",
        )

    console.print(table)

    table = Table(show_header=True, header_style="bold magenta")

    table.add_column("Compression Ratio")
    table.add_column("Raw Identifier Splitting Score")
    table.add_column("Identifier Splitting Score")

    table.add_row(
        str(result.metrics.compression),
        str(result.metrics.raw_identifier_splitting_score),
        str(result.metrics.identifier_splitting_score),
    )

    console.print(table)
