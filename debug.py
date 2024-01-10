# from itertools import islice
# from typing import Iterable

from itertools import islice
from typing import Any, Iterable, Iterator, List

import tiktoken
from datasets import Dataset, load_dataset
from tqdm import tqdm


def batches(it: Iterable[Any], size: int) -> Iterator[List[Any]]:
    """Yield successive batches of size 'batch_size' from 'iterable'."""
    it = iter(it)
    return iter(lambda: list(islice(it, size)), [])


enc = tiktoken.encoding_for_model("gpt-4")

the_stack_smol: Dataset = load_dataset("bigcode/the-stack-smol", split="train")  # type: ignore

print(f"Dataset Size: {the_stack_smol.num_rows}")

total_content_bytes = 0
total_content_tokens = 0

for batch in tqdm(batches(the_stack_smol, 100)):
    contents = [sample["content"] for sample in batch]  # type: ignore

    ids = enc.encode_ordinary_batch(contents)

    total_content_bytes += sum(
        len(content.encode("utf-8", errors="ignore")) for content in contents
    )
    total_content_tokens += sum(len(i) for i in ids)

print(f"Total Content Bytes: {total_content_bytes}")
print(f"Total Content Tokens: {total_content_tokens}")
