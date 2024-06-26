import json
from typing import Dict, List

from datasets import load_dataset
from tiktoken import encoding_for_model
from tqdm import tqdm
from transformers import AutoTokenizer

HF_TOKENIZER_NAMES = [
    "replit/replit-code-v1_5-3b",
    "stabilityai/stable-code-3b",
    "codellama/CodeLlama-7b-hf",
]

HF_TOKENIZERS = [
    AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    for model_name in HF_TOKENIZER_NAMES
]

OPENAI_TOKENIZER_NAMES = [
    "code-cushman-001",
    "gpt-4",
]

OPENAI_TOKENIZERS = [
    encoding_for_model(model_name) for model_name in OPENAI_TOKENIZER_NAMES
]


def batched_dataset(batch_size):
    batch = []
    for sample in tqdm(load_dataset("bigcode/the-stack-smol", split="train")):
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


# For each tokeniser, mapping from token ID to number of occurrences for that
# token.
r: Dict[str, Dict[int, int]] = {
    name: {} for name in HF_TOKENIZER_NAMES + OPENAI_TOKENIZER_NAMES
}

for samples in batched_dataset(256):
    contents = [sample["content"] for sample in samples]

    for name, tokenizer in zip(HF_TOKENIZER_NAMES, HF_TOKENIZERS):
        ids: List[List[int]] = tokenizer.batch_encode_plus(
            contents, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]  # type: ignore
        for s in ids:
            for id in s:
                if id not in r[name]:
                    r[name][id] = 0
                r[name][id] += 1

    for name, tokenizer in zip(OPENAI_TOKENIZER_NAMES, OPENAI_TOKENIZERS):
        ids = tokenizer.encode_ordinary_batch(contents)
        for s in ids:
            for id in s:
                if id not in r[name]:
                    r[name][id] = 0
                r[name][id] += 1

print(
    json.dumps(
        r,
        indent=2,
    )
)

with open("results/token-frequencies.json", "w") as f:
    json.dump(r, f, indent=2)
