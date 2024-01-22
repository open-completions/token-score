import logging
import os
import sys
from multiprocessing import Pool, cpu_count
from typing import Iterator, List, Optional, Tuple

import tiktoken
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from token_score import (
    SUPPORTED_LANGUAGES,
    Document,
    TokenScoreMetrics,
    compute_token_score,
    huggingface_tokenizer,
    tiktoken_tokenizer,
)

if len(sys.argv) < 4:
    print("Usage: python evaluate_the_stack.py <lib> <model> <dataset> <outdir>")
    sys.exit(1)

lib = sys.argv[1]
model = sys.argv[2]
dataset = sys.argv[3]
outdir = sys.argv[4]

assert lib in ["hf", "tiktoken"]
assert dataset in ["bigcode/the-stack-smol", "bigcode/the-stack-smol-xs"]

is_full_run = dataset == "bigcode/the-stack-smol"

tokenizer = (
    AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if lib == "hf"
    else tiktoken.encoding_for_model(model)
)


def the_stack_to_documents(datasets: List[Dataset]) -> Iterator[Document]:
    for ds in datasets:
        for sample in ds:
            lang = sample["lang"].lower()  # type: ignore

            yield Document(
                lang=lang,  # type: ignore
                content=sample["content"].encode("utf-8", errors="ignore"),  # type: ignore
            )


def worker_process(
    doc: Document,
) -> Tuple[Optional[TokenScoreMetrics], Optional[str], Optional[Exception]]:
    try:
        tokens = (
            tiktoken_tokenizer(tokenizer, doc)  # type: ignore
            if lib == "tiktoken"
            else huggingface_tokenizer(tokenizer, doc)  # type: ignore
        )
        score = compute_token_score(
            doc, tokens, return_token_span_score=not is_full_run
        )
        return score.metrics, doc.lang, None

    except Exception as e:
        logging.error(f"Failed to compute token score: {e.__class__.__name__} {e}")
        return None, None, e


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    logging.info(f"Computing token score for {lib}/{model} over {dataset}")

    if is_full_run:
        logging.info("Omitting token span score for full run")

    the_stack_smol = (
        (
            [
                load_dataset(dataset, data_dir=f"data/{lang}", split="train")
                for lang in SUPPORTED_LANGUAGES
            ]
        )
        if is_full_run
        else (
            [
                load_dataset(dataset, lang, split="train", trust_remote_code=True)
                for lang in SUPPORTED_LANGUAGES
            ]
        )
    )

    total = sum([len(ds) for ds in the_stack_smol])  # type: ignore

    logging.info(f"Computing token score for {total} documents")

    def outfile_for_lang(lang: str) -> str:
        return f"{outdir}/{lang}/{lib}-{model.replace("/", "-")}/{dataset.replace("/", "-")}.csv"

    [
        os.makedirs(os.path.dirname(outfile_for_lang(lang)), exist_ok=True)
        for lang in SUPPORTED_LANGUAGES
    ]

    files = {lang: open(outfile_for_lang(lang), "a") for lang in SUPPORTED_LANGUAGES}

    for file in files.values():
        file.write(
            "total_tokens,total_bytes,compression,token_span_score,raw_identifier_splitting_score,identifier_splitting_score,identifier_fertility\n"
        )

    with Pool(cpu_count()) as pool:
        tasks = (doc for doc in the_stack_to_documents(the_stack_smol))  # type: ignore

        for m, lang, e in tqdm(pool.imap_unordered(worker_process, tasks), total=total):
            if e is not None:
                logging.error(f"Failed to compute token score: {e}")
                continue

            if m is not None and lang is not None:
                files[lang].write(
                    f"{m.total_tokens},{m.total_bytes},{m.compression},{m.token_span_score},{m.raw_identifier_splitting_score},{m.identifier_splitting_score},{m.identifier_fertility}\n"
                )
