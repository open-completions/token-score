import tiktoken
from datasets import Dataset, load_dataset
from tqdm import tqdm

enc = tiktoken.encoding_for_model("gpt-4")

the_stack_smol: Dataset = load_dataset("bigcode/the-stack-smol", split="train")  # type: ignore

print(f"Dataset Size: {the_stack_smol.num_rows}")

total_content_bytes = 0
total_content_tokens = 0

for sample in tqdm(the_stack_smol):
    content: str = sample["content"]  # type: ignore

    ids = enc.encode_ordinary(content)

    enc_content = content.encode("utf-8", errors="ignore")
    total_content_bytes += len(enc_content)
    total_content_tokens += len(ids)

print(f"Total Content Bytes: {total_content_bytes}")
print(f"Total Content Tokens: {total_content_tokens}")
