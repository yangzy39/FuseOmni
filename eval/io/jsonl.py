"""
IO module for reading/writing JSONL files.
"""

import json
from pathlib import Path
from typing import Iterator, Any


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    """
    Write records to a JSONL file.
    
    Args:
        path: Output file path
        records: List of dictionaries to write
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, records: list[dict]) -> None:
    """
    Append records to a JSONL file.
    
    Args:
        path: Output file path
        records: List of dictionaries to append
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """
    Read records from a JSONL file.
    
    Args:
        path: Input file path
        
    Yields:
        Dictionary for each line
    """
    path = Path(path)
    
    if not path.exists():
        return
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: str | Path) -> list[dict]:
    """
    Load all records from a JSONL file into memory.
    
    Args:
        path: Input file path
        
    Returns:
        List of dictionaries
    """
    return list(read_jsonl(path))
