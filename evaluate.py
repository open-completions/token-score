import argparse
import json
import os
from collections import deque
from typing import Callable, List, Literal, Tuple

from datasets import Dataset, load_dataset
from tqdm import tqdm

MODELS = [
    "phi-2",
    "gpt-4",
    "codex",
    "mistral",
    "codellama",
    "replit",
]

ModelName = Literal[
    "phi-2",
    "gpt-4",
    "codex",
    "mistral",
    "codellama",
    "replit",
]

MODEL_TO_MODEL_SLUG = {
    "phi-2": "microsoft/phi-2",
    "gpt-4": "gpt-4",
    "codex": "code-cushman-001",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "codellama": "codellama/CodeLlama-7b-hf",
    "replit": "replit/replit-code-v1_5-3b",
}


def batch(iterable, n=1):
    it = iter(iterable)
    q = deque(maxlen=n)
    for item in it:
        q.append(item)
        if len(q) == n:
            yield list(q)
    if q:  # yield any remaining items
        yield list(q)


def get_tokenizer_for_model(
    name: ModelName,
) -> Tuple[Callable[[str], List[bytes]], int, bool]:
    if name in ["phi-2", "mistral", "codellama", "replit"]:
        # Can use AutoTokenizer from HuggingFace
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_MODEL_SLUG[name])

        def hf_tokenize_fn(text: str):
            return [
                tok.encode("utf-8")
                for tok in tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
            ]

        return hf_tokenize_fn, len(tokenizer), tokenizer.is_fast

    if name in ["gpt-4", "codex", "codex-2"]:
        # Can use tiktoken
        import tiktoken

        enc = tiktoken.encoding_for_model(MODEL_TO_MODEL_SLUG[name])

        def tokenize_fn(text: str):
            return [enc.decode_single_token_bytes(t) for t in enc.encode_ordinary(text)]

        return tokenize_fn, enc.max_token_value + 1, True

    raise ValueError(f"Unknown model name: {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=MODELS,
        required=True,
    )
    args = parser.parse_args()

    batch_tokenize_fn, vocab_size, is_fast = get_tokenizer_for_model(args.model)

    the_stack_smol: Dataset = load_dataset("bigcode/the-stack-smol", split="train")  # type: ignore

    print(f"Model: {args.model}")
    print(f"Vocabulary Size: {vocab_size}")
    print(
        f"Example: {[b.decode(errors="ignore") for b in batch_tokenize_fn(f"# 你好！我叫{args.model}\nfor i in range(10):\n    do_complicated_stuff(complex_data[i])")]}"
    )
    print(f"Dataset Size: {the_stack_smol.num_rows}")
    print(f"Is Fast: {is_fast}")

    tokens_by_language = {}
    bytes_per_language = {}

    for samples in tqdm(
        batch(the_stack_smol, n=256), total=the_stack_smol.num_rows // 256
    ):
        contents = [s["content"] for s in samples]  # type: ignore
        langs = [s["lang"] for s in samples]  # type: ignore

        token_batches = batch_tokenize_fn(contents)

        for lang, tokens, contents in zip(langs, token_batches, contents):
            if lang not in tokens_by_language:
                tokens_by_language[lang] = 0
                bytes_per_language[lang] = 0

            bytes_len = len(contents.encode("utf-8"))

            tokens_by_language[lang] += len(tokens)
            bytes_per_language[lang] += bytes_len

    print("Tokens by Language:")
    for lang, tokens in sorted(tokens_by_language.items(), key=lambda x: x[1]):
        print(f"  {lang}: {tokens}")
    print("Bytes by Language:")
    for lang, bytes in sorted(bytes_per_language.items(), key=lambda x: x[1]):
        print(f"  {lang}: {bytes}")

    print("Done!")

    # Write results to a JSON file
    os.makedirs("results", exist_ok=True)
    with open(f"results/{args.model}.json", "w") as f:
        json.dump(
            {
                "model": args.model,
                "vocab_size": vocab_size,
                "dataset_size": the_stack_smol.num_rows,
                "tokens_by_language": tokens_by_language,
                "bytes_per_language": bytes_per_language,
            },
            f,
            indent=2,
        )
