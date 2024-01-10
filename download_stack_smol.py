"""
Utility script for pre-loading `bigcode/the-stack-smol` from HuggingFace.
"""

from datasets import Dataset, load_dataset

the_stack_smol: Dataset = load_dataset("bigcode/the-stack-smol", split="train")  # type: ignore

print(the_stack_smol.column_names)
