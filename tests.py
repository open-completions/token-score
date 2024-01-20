from pprint import pprint

from transformers import AutoTokenizer

from token_score import Document, huggingface_tokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "codellama/CodeLlama-7b-hf", trust_remote_code=True
)

seqs = [b"def foo():\n    print('hello world')\n", "你叫什么名字？".encode("utf-8")]

for seq in seqs:
    pprint(
        [
            tokenizer.decode(t)
            for t in tokenizer.encode(seq.decode("utf-8"), add_special_tokens=False)
        ]
    )
    tokens = huggingface_tokenizer(tokenizer, Document(lang="python", content=seq))
    pprint(tokens)
    pprint([seq[t.range[0] : t.range[1]] for t in tokens])
