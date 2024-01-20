import logging
import sys
from multiprocessing import Pool, cpu_count
from typing import Iterator, List, Tuple, Union

import tiktoken
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from token_score import (
    SUPPORTED_LANGUAGES,
    Document,
    TokenScore,
    TokenScoreMetrics,
    compute_token_score,
    huggingface_tokenizer,
)

HF_TOKENIZER = AutoTokenizer.from_pretrained(
    "codellama/CodeLlama-7b-hf", trust_remote_code=True
)

OPENAI_TOKENIZER = tiktoken.encoding_for_model("code-cushman-001")


def the_stack_to_documents(datasets: List[Dataset]) -> Iterator[Document]:
    for ds in datasets:
        for sample in ds:
            lang = sample["lang"].lower()  # type: ignore

            if lang not in SUPPORTED_LANGUAGES:  # type: ignore
                continue

            yield Document(
                lang=lang,  # type: ignore
                content=sample["content"].encode("utf-8", errors="ignore"),  # type: ignore
            )


def worker_process(doc: Document) -> Union[None, Tuple[TokenScoreMetrics, str]]:
    try:
        if len(doc.content) > 256 * 1024:
            logging.warning(f"Skipping because too long: {len(doc.content)} bytes")
            return None

        # tokens = tiktoken_tokenizer(OPENAI_TOKENIZER, doc)
        tokens = huggingface_tokenizer(HF_TOKENIZER, doc)

        r = compute_token_score(doc, tokens)
        return r.metrics, doc.lang

    except Exception as e:
        logging.error(f"Failed to compute token score: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_the_stack.py <outfile>")
        sys.exit(1)

    ds = "bigcode/the-stack-smol-xs"

    the_stack_smol_py: Dataset = load_dataset(
        ds,
        "python",
        split="train",
        trust_remote_code=True,
    )  # type: ignore
    the_stack_smol_go: Dataset = load_dataset(
        ds, "go", split="train", trust_remote_code=True
    )  # type: ignore
    the_stack_smol_java: Dataset = load_dataset(
        ds, "java", split="train", trust_remote_code=True
    )  # type: ignore
    the_stack_smol_javascript: Dataset = load_dataset(
        ds,
        "javascript",
        split="train",
        trust_remote_code=True,
    )  # type: ignore
    the_stack_smol_cpp: Dataset = load_dataset(
        ds, "c++", split="train", trust_remote_code=True
    )  # type: ignore

    score = TokenScore(
        metrics={
            lang: TokenScoreMetrics(
                compression=0,
                total_bytes=0,
                total_tokens=0,
                token_span_score=0,
                identifier_fertility=0,
                identifier_splitting_score=0,
                raw_identifier_splitting_score=0,
            )
            for lang in SUPPORTED_LANGUAGES
        },
        total_bytes=0,
        total_tokens=0,
    )

    with Pool(cpu_count()) as pool:
        tasks = (
            doc
            for doc in the_stack_to_documents(
                [
                    the_stack_smol_py,
                    the_stack_smol_go,
                    the_stack_smol_java,
                    the_stack_smol_javascript,
                    the_stack_smol_cpp,
                ]
            )
        )

        for r in tqdm(pool.imap_unordered(worker_process, tasks), total=500):
            if r is not None:
                score.add(r[0], r[1])

    print(score.model_dump_json())

    with open(f"{sys.argv[1]}", "w") as f:
        f.write(score.model_dump_json())
