import logging
from multiprocessing import Pool, cpu_count
from typing import Iterator, List, Tuple, Union

import tiktoken
from datasets import Dataset, load_dataset
from tqdm import tqdm

from token_score import (
    SUPPORTED_LANGUAGES,
    Document,
    TokenScore,
    TokenScoreMetrics,
    compute_token_score,
    tiktoken_tokenizer,
)


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
    enc = tiktoken.encoding_for_model("gpt-4")

    try:
        if len(doc.content) > 256 * 1024:
            logging.warning(f"Skipping because too long: {len(doc.content)} bytes")
            return None

        tokens = tiktoken_tokenizer(enc, doc)
        r = compute_token_score(doc, tokens)
        return r.metrics, doc.lang

    except Exception as e:
        logging.error(f"Failed to compute token score: {e}")
        return None


if __name__ == "__main__":
    ds = "bigcode/the-stack-smol"

    the_stack_smol_py: Dataset = load_dataset(
        ds,
        data_dir="data/python",
        split="train",
        trust_remote_code=True,
    )  # type: ignore
    the_stack_smol_go: Dataset = load_dataset(
        ds, data_dir="data/go", split="train", trust_remote_code=True
    )  # type: ignore
    the_stack_smol_java: Dataset = load_dataset(
        ds, data_dir="data/java", split="train", trust_remote_code=True
    )  # type: ignore
    the_stack_smol_javascript: Dataset = load_dataset(
        ds,
        data_dir="data/javascript",
        split="train",
        trust_remote_code=True,
    )  # type: ignore
    the_stack_smol_cpp: Dataset = load_dataset(
        ds, data_dir="data/c++", split="train", trust_remote_code=True
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

        for r in tqdm(pool.imap_unordered(worker_process, tasks), total=50000):
            if r is not None:
                score.add(r[0], r[1])

    print(score.model_dump_json())

    with open("the_stack_smol_metrics.json", "w") as f:
        f.write(score.model_dump_json())
