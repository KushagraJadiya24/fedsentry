"""Prepare the three locked non-IID JSONL client partitions.

Owner: Kashish( Data + Client Wiring )

Day-0 scaffold only.

Final implementation requirements:
- Generate:
    data/partitions/client_0.jsonl
    data/partitions/client_1.jsonl
    data/partitions/client_2.jsonl
- Each line must be:
    {"instruction": "...", "response": "..."}
- The three shards must be non-IID.
- Document the source dataset and split method in this module docstring.

Source dataset: Stanford Alpaca (tatsu-lab/stanford_alpaca), 52K instruction/
response pairs generated via OpenAI text-davinci-003 using the Self-Instruct
method. Licensed CC BY NC 4.0 (research use only). Downloaded from:
https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
Currently loading a 300-example slice during development (see
load_alpaca_examples()).

Run from the repository root:
    python data/prepare_data.py
"""
import json
from pathlib import Path



PARTITIONS_DIR = Path(__file__).resolve().parent / "partitions"
ALPACA_FILE = Path(__file__).resolve().parent / "alpaca_data.json"

def load_alpaca_examples(limit:int=300)-> list[dict]:
    with open(ALPACA_FILE,"r",encoding="utf-8") as f:
        all_examples =json.load(f)
    return all_examples[:limit]
    


def main() -> None:
    examples=load_alpaca_examples()
    print(f"Loaded {len(examples)} examples")
    print(examples[0])


if __name__ == "__main__":
    main()
