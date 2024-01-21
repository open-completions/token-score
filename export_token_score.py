"""
Compute the average for all the metrics produced by tokeniser evaluation for a
single language
"""

import argparse
import csv
import json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", "-m", required=True, help="The model for which to compute the aggregate metrics for")
    p.add_argument("--dataset", "-d", required=True, help="The dataset for which to compute the aggregate metrics for")
    args = p.parse_args()

    for lang in ["java", "python", "go", "javascript", "c++"]:
        with open(f"results/{lang}/{args.model}/{args.dataset}.csv", "r") as f:
            reader = csv.DictReader(f)
            data = list(reader)

            # Format:
            # total_tokens,total_bytes,compression,token_span_score,raw_identifier_splitting_score,identifier_splitting_score,identifier_fertility

            total_tokens = 0
            total_bytes = 0
            token_span_score = 0
            raw_identifier_splitting_score = 0
            identifier_splitting_score = 0
            identifier_fertility = 0

            for row in data:
                total_tokens += int(row["total_tokens"])
                total_bytes += int(row["total_bytes"])
                token_span_score += float(row["token_span_score"])
                raw_identifier_splitting_score += float(row["raw_identifier_splitting_score"])
                identifier_splitting_score += float(row["identifier_splitting_score"])
                identifier_fertility += float(row["identifier_fertility"])

            token_span_score /= len(data)
            raw_identifier_splitting_score /= len(data)
            identifier_splitting_score /= len(data)
            identifier_fertility /= len(data)

            compression = total_bytes / total_tokens

        with open(f"results/{lang}/{args.model}/{args.dataset}.json", "w") as o:
            json.dump({
                "total_tokens": total_tokens,
                "total_bytes": total_bytes,
                "compression": compression,
                "token_span_score": token_span_score,
                "raw_identifier_splitting_score": raw_identifier_splitting_score,
                "identifier_splitting_score": identifier_splitting_score,
                "identifier_fertility": identifier_fertility
            }, o)

        print(f"-------- {lang} {args.model} {args.dataset} --------")
        for metric_display_name, value in [
            ("Compression", compression),
            ("Identifier Splitting Score", identifier_splitting_score),
            ("Identifier Fertility", identifier_fertility),
            ("Token Span Score", token_span_score),
        ]:
            print(f"{metric_display_name}: {round(value, ndigits=2)}")
