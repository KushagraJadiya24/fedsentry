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
    """Loads the first `limit` examples from alpaca_data.json."""
    with open(ALPACA_FILE,"r",encoding="utf-8") as f:
        all_examples =json.load(f)
    return all_examples[:limit]

def convert_to_locked_schema(examples:dict)-> dict:
    """Converts one raw Alpaca example ({instruction, input, output}) into
    the locked FedSentry schema ({instruction, response})."""
    instruction=examples["instruction"]
    if examples["input"]:
        instruction=instruction + " " + examples["input"]
    response=examples["output"]
    return {"instruction": instruction, "response": response}
    


def main() -> None:
    examples=load_alpaca_examples()
    print(f"Loaded {len(examples)} examples")
    print(examples[0])
    converted = convert_to_locked_schema(examples[0])
    print("Example 0:", converted)
    converted5 = convert_to_locked_schema(examples[5])
    print("Example 5:", converted5)


if __name__ == "__main__":
    main()
