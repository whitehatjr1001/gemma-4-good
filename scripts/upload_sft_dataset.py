from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from gemma_health.data.sft import load_sft_jsonl_sample, validate_sft_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload an SFT JSONL file to a Hugging Face dataset repo.")
    parser.add_argument("--input", default="data/processed/sft/train.jsonl")
    parser.add_argument("--repo-id", required=True, help="Example: username/gemma-health-telugu-sft")
    parser.add_argument("--split", default="train")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--upload", action="store_true", help="Actually upload. Without this, only validates.")
    args = parser.parse_args()

    input_path = Path(args.input)
    row_count = validate_sft_jsonl(input_path)
    sample = load_sft_jsonl_sample(input_path, limit=1)[0]

    print(f"validated {row_count} rows from {input_path}")
    print(f"sample source={sample['source']} variant={sample['variant']}")
    print(f"streaming train command: load_dataset('{args.repo_id}', split='{args.split}', streaming=True)")

    if not args.upload:
        print("dry run only; pass --upload to publish to Hugging Face")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as err:
        raise RuntimeError("Install huggingface_hub or datasets before uploading to Hugging Face") from err

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_or_fileobj=input_path,
        path_in_repo=f"data/{args.split}.jsonl",
    )
    with TemporaryDirectory() as temp_dir:
        readme_path = Path(temp_dir) / "README.md"
        readme_path.write_text(_dataset_card(args.repo_id, args.split, row_count), encoding="utf-8")
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
        )
    print(f"uploaded {row_count} rows to {args.repo_id}")


def _dataset_card(repo_id: str, split: str, row_count: int) -> str:
    return f"""---
task_categories:
- text-generation
language:
- te
tags:
- sft
- telugu
- medical
---

# Gemma Health Telugu SFT

Rows: {row_count}

Each row contains:

- `messages`: TRL/Unsloth conversational SFT format.
- `text`: plain serialized chat text fallback.
- `source`, `variant`, `prompt`, `response`: traceability fields.

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}", split="{split}", streaming=True)
```
"""


if __name__ == "__main__":
    main()
