from typing import Iterator

import tiktoken
from datasets import Dataset, load_dataset
from tqdm import tqdm

from token_score import (
    SUPPORTED_LANGUAGES,
    Document,
    collect_identifiers,
)


def the_stack_to_documents(ds: Dataset) -> Iterator[Document]:
    for sample in ds:
        lang = sample["lang"].lower()  # type: ignore

        if lang not in SUPPORTED_LANGUAGES:  # type: ignore
            continue

        yield Document(
            lang=lang,  # type: ignore
            content=sample["content"].encode("utf-8", errors="ignore"),  # type: ignore
        )


if __name__ == "__main__":
    the_stack_smol: Dataset = load_dataset("bigcode/the-stack-smol", split="train")  # type: ignore

    enc = tiktoken.encoding_for_model("gpt-4")

    for doc in tqdm(the_stack_to_documents(the_stack_smol), total=50000):
        tree = doc.parse()

        # semantic_tokens = collect_semantic_tokens(tree, doc.content)

        identifiers = collect_identifiers(tree, doc)

        # tokens = tiktoken_tokenizer(enc, doc)
        # result = compute_token_score(doc, tokens)
