import json

import tiktoken

with open("token_frequencies.json") as f:
    r = json.load(f)

for (name, d) in r.items():
    if len(d) == 0:
        continue

    enc = tiktoken.encoding_for_model(name)

    total_tokens = sum(d.values())
    print(f"{name}: {total_tokens} tokens")
    print(f"{name}: {len(d)} unique tokens")
    print(f"{name}: {total_tokens / len(d)} average occurrences per token")
    print(f"{name}: {sum(1 for v in d.values() if v == 1)} tokens that only occur once")
    print(f"{name}: {sum(1 for v in d.values() if v == 2)} tokens that only occur twice")
    print(f"{name}: {sum(1 for v in d.values() if v == 3)} tokens that only occur thrice")
    print(f"{name}: {sum(1 for v in d.values() if v == 4)} tokens that only occur four times")

    # Example tokens that occur only once:
    print(f"{name}: {[enc.decode_single_token_bytes(int(d)) for d in sorted(d, key=lambda k: d[k])[:10]]}")

