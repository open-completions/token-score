import json

from tiktoken import encoding_for_model
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


with open("results/token-frequencies.json", "r") as f:
    r = json.load(f)
    # Change all the keys from str to int.
    for name in r:
        r[name] = {int(k): v for k, v in r[name].items()}


# Prefill with all tokens in the vocabulary, so that we can see which tokens
# never occur.
for name, tokenizer in zip(HF_TOKENIZER_NAMES, HF_TOKENIZERS):
    for id in range(tokenizer.vocab_size):
        if id not in r[name] and id not in tokenizer.all_special_ids:
            r[name][id] = 0
for name, tokenizer in zip(OPENAI_TOKENIZER_NAMES, OPENAI_TOKENIZERS):
    for id in range(tokenizer.max_token_value + 1):
        if id not in r[name] and id not in tokenizer._special_tokens.values():
            r[name][id] = 0

with open("results/token-frequencies-zero-occurence.json", "w") as f:
    json.dump(r, f, indent=2)
