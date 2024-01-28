import json
import logging

from tiktoken import encoding_for_model
from transformers import AutoTokenizer

HF_TOKENIZER_NAMES = [
    "replit/replit-code-v1_5-3b",
    "stabilityai/stable-code-3b",
    "codellama/CodeLlama-7b-hf",
]

HF_TOKENIZERS = {
    model_name: AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    for model_name in HF_TOKENIZER_NAMES
}

OPENAI_TOKENIZER_NAMES = [
    "code-cushman-001",
    "gpt-4",
]

OPENAI_TOKENIZERS = {
    model_name: encoding_for_model(model_name) for model_name in OPENAI_TOKENIZER_NAMES
}


# Step 1: Load the data
with open("results/token-frequencies-zero-occurence.json", "r") as file:
    data = json.load(file)

# Sort the token frequencies in descending order
for tokenizer_name, frequencies in data.items():
    data[tokenizer_name] = {
        k: v
        for k, v in sorted(frequencies.items(), key=lambda item: item[1], reverse=True)
    }

r = {}

for tokenizer_name, frequencies in data.items():
    r[tokenizer_name] = []

    for id, frequency in list(reversed(list(frequencies.items()))):
        token_str = None

        if tokenizer_name in HF_TOKENIZERS:
            token_str = HF_TOKENIZERS[tokenizer_name]._convert_id_to_token(int(id))

        if tokenizer_name in OPENAI_TOKENIZERS:
            try:
                token_str = (
                    OPENAI_TOKENIZERS[tokenizer_name]
                    .decode_single_token_bytes(int(id))
                    .decode("utf-8", errors="ignore")
                )
            except Exception:
                logging.error(f"Failed to decode token {id} for {tokenizer_name}")
                token_str = "UNK"

        r[tokenizer_name].append((token_str, frequency))


with open("results/token-frequencies-top.json", "w") as file:
    json.dump(r, file, indent=2)
