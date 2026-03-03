from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable


class MetricLogger:
    def __init__(self, csv_path: Path, jsonl_path: Path) -> None:
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self._header_written = csv_path.exists() and csv_path.stat().st_size > 0

    def log(self, row: Dict[str, float]) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.jsonl_path, "a", encoding="utf-8") as jf:
            jf.write(json.dumps(row) + "\n")

        with open(self.csv_path, "a", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_markdown(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
