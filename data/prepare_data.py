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

Split method: each converted example is tagged by topic via keyword matching
(tag_topic()). Matched examples (science / history_geography / coding_tech)
create the non-IID skew -- each client gets ALL of its home topic's matched
examples. Since a general dataset like Alpaca has many examples that don't
cleanly fit 3 narrow topics, unmatched examples are pooled and split evenly
across all 3 clients as shared "general" data, rounding out each client's
dataset size without diluting the topic skew.

Run from the repository root:
    python data/prepare_data.py
"""
import json
import random
from pathlib import Path


PARTITIONS_DIR = Path(__file__).resolve().parent / "partitions"
ALPACA_FILE = Path(__file__).resolve().parent / "alpaca_data.json"

TOPIC_KEYWORDS = {
    "science": ["photosynthesis", "atom", "cell", "gravity", "energy", "biology",
                "physics", "chemical", "planet", "organism", "molecule", "force",
                "species", "disease", "medicine", "health", "body", "nutrition",
                "climate", "weather", "environment", "animal", "plant"],
    "history_geography": ["capital", "war", "country", "president", "century",
                           "continent", "river", "mountain", "empire", "revolution",
                           "julius caesar", "king", "queen", "ancient", "civilization",
                           "history", "historical", "nation", "border", "population"],
    "coding_tech": ["python", "code", "function", "algorithm", "programming",
                     "software", "variable", "loop", "database", "array", "class",
                     "computer", "internet", "website", "app", "technology", "digital"],
}

CLIENT_HOME_TOPIC = {0: "science", 1: "history_geography", 2: "coding_tech"}


def load_alpaca_examples(limit: int = 300) -> list[dict]:
    """Loads the first `limit` examples from alpaca_data.json."""
    with open(ALPACA_FILE, "r", encoding="utf-8") as f:
        all_examples = json.load(f)
    return all_examples[:limit]


def convert_to_locked_schema(examples: dict) -> dict:
    """Converts one raw Alpaca example ({instruction, input, output}) into
    the locked FedSentry schema ({instruction, response})."""
    instruction = examples["instruction"]
    if examples["input"]:
        instruction = instruction + " " + examples["input"]
    response = examples["output"]
    return {"instruction": instruction, "response": response}


def tag_topic(instruction: str) -> str:
    """Assigns one of the 3 topics to an instruction, via keyword matching."""
    text = instruction.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return topic
    return "unmatched"  # visible instead of hidden inside "science"


def group_by_topic(examples: list[dict]) -> dict[str, list[dict]]:
    """Groups converted examples into topic buckets using tag_topic()."""
    groups = {"science": [], "history_geography": [], "coding_tech": [], "unmatched": []}
    for ex in examples:
        topic = tag_topic(ex["instruction"])
        groups[topic].append(ex)
    return groups


def build_non_iid_split(examples: list[dict], rng: random.Random) -> dict[int, list[dict]]:
    """Each client gets most of its 'home' topic's matched examples, plus an
    even share of the unmatched/general pool to round out its dataset."""
    groups = group_by_topic(examples)
    for items in groups.values():
        rng.shuffle(items)

    unmatched_pool = groups["unmatched"]
    third = len(unmatched_pool) // 3

    client_data = {0: [], 1: [], 2: []}
    for client_id, home_topic in CLIENT_HOME_TOPIC.items():
        client_data[client_id].extend(groups[home_topic])          # skewed part
        start = client_id * third
        end = start + third
        client_data[client_id].extend(unmatched_pool[start:end])   # even filler
        rng.shuffle(client_data[client_id])

    return client_data


def validate_schema(examples: list[dict]) -> None:
    """Checks every example has exactly the two locked keys, both non-empty."""
    for i, ex in enumerate(examples):
        assert set(ex.keys()) == {"instruction", "response"}, (
            f"line {i} has wrong keys: {list(ex.keys())}"
        )
        assert ex["instruction"].strip(), f"line {i} has an empty instruction"
        assert ex["response"].strip(), f"line {i} has an empty response"


def write_partitions(client_data: dict[int, list[dict]]) -> None:
    """Validates and writes each client's examples to its locked .jsonl file."""
    PARTITIONS_DIR.mkdir(parents=True, exist_ok=True)
    for client_id, items in client_data.items():
        validate_schema(items)
        out_path = PARTITIONS_DIR / f"client_{client_id}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in items:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {len(items)} examples -> {out_path}")


def main() -> None:
    examples = load_alpaca_examples()
    print(f"Loaded {len(examples)} examples")

    converted_all = [convert_to_locked_schema(ex) for ex in examples]
    converted_all = [ex for ex in converted_all if ex["instruction"].strip() and ex["response"].strip()]
    print(f"{len(converted_all)} examples remain after removing empty fields")

    rng = random.Random(42)
    client_data = build_non_iid_split(converted_all, rng)
    write_partitions(client_data)


if __name__ == "__main__":
    main()