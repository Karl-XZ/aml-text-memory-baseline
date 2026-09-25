"""Local retrieval diagnostic on the independently public LoCoMo data.

This script never loads AML evaluation memories or gold answers into Search.
Answers and evidence labels are used only after retrieval for measurement.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import server
import experimental_option_route


def session_names(conversation: dict) -> list[str]:
    return sorted(
        (key for key in conversation if key.startswith("session_") and key[8:].isdigit()),
        key=lambda name: int(name.split("_")[1]),
    )


def index_sample(db: Path, sample: dict) -> tuple[str, dict[str, str]]:
    user_id = "public-locomo:" + sample["sample_id"]
    source_ids = {}
    for session in session_names(sample["conversation"]):
        date = sample["conversation"].get(session + "_date_time", "")
        turns = sample["conversation"][session]
        for start in range(0, len(turns), 20):
            chunk = turns[start:start + 20]
            request_id = f"public-locomo:{sample['sample_id']}:{session}:{start // 20}"
            messages = []
            for offset, turn in enumerate(chunk):
                messages.append({"role": "user", "content": f"{date} | {turn['speaker']}: {turn['text']}"})
                source_ids[turn["dia_id"]] = server.memory_id(request_id, offset)
            server.add(db, {"request_id": request_id, "messages": messages,
                            "user_id": user_id,
                            "session_id": f"public-locomo:{sample['sample_id']}:{session}"})
    return user_id, source_ids


def source_ids_for_result(db: Path, result: dict, all_ids: dict[str, str]) -> list[str]:
    if "_source_ids" in result:
        return result["_source_ids"]
    return [result["id"]]


def evaluate(sample: dict, questions: list[dict], top_k: int, variant: str,
             mc_options: bool) -> list[dict]:
    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "memory.sqlite3"
        server.initialize(db)
        user_id, source_ids = index_sample(db, sample)
        rows = []
        for question in questions:
            if mc_options and not question.get("answer"):
                continue
            refs = [part.strip() for item in question.get("evidence", [])
                    for part in item.split(";")]
            if not refs or not all(ref in source_ids for ref in refs):
                continue
            gold = {source_ids[ref] for ref in refs}
            request = {"query": question["question"], "user_id": user_id,
                       "top_k": top_k}
            if mc_options:
                distractors = [item["answer"] for item in questions
                               if item is not question and item.get("answer")
                               and item["answer"] != question["answer"]]
                request["options"] = [str(question["answer"])] + [str(x) for x in distractors[:3]]
            retrieval = (experimental_option_route.search_option_evidence if variant == "options"
                         else server.search_spans)
            retrieved = retrieval(db, request, radius=5,
                                  include_source_ids=True)["data"]
            positions = [rank for rank, item in enumerate(retrieved, 1)
                         if gold.intersection(item.get("_source_ids", [item["id"]]))]
            seen = {source_id for item in retrieved
                    for source_id in item.get("_source_ids", [item["id"]])}
            rows.append({"sample_id": sample["sample_id"], "question": question["question"],
                         "any": bool(gold & seen), "all": gold <= seen,
                         "first_gold_rank": min(positions, default=None),
                         "gold_in_first_10": any(position <= 10 for position in positions),
                         "results": len(retrieved),
                         "chars_first_20": sum(len(item["content"]) for item in retrieved[:20]),
                         "chars_all": sum(len(item["content"]) for item in retrieved)})
        return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--variant", choices=("baseline", "options"), default="baseline")
    parser.add_argument("--mc-options", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    samples = {item["sample_id"]: item for item in
               json.loads(args.data.read_text(encoding="utf-8"))}
    if args.all_questions:
        selected = {sample_id: sample["qa"] for sample_id, sample in samples.items()}
    else:
        if args.manifest is None:
            parser.error("--manifest is required unless --all-questions is selected")
        selected = {}
        for item in json.loads(args.manifest.read_text(encoding="utf-8"))["questions"]:
            selected.setdefault(item["sample_id"], []).append(item)
    rows = []
    for sample_id, questions in selected.items():
        rows.extend(evaluate(samples[sample_id], questions, args.top_k, args.variant,
                             args.mc_options))
    summary = {"questions": len(rows),
               "any": sum(row["any"] for row in rows),
               "all": sum(row["all"] for row in rows),
               "gold_in_first_10": sum(row["gold_in_first_10"] for row in rows),
               "mean_chars_first_20": round(sum(row["chars_first_20"] for row in rows) / len(rows)),
               "mean_chars_all": round(sum(row["chars_all"] for row in rows) / len(rows))}
    report = {"variant": args.variant, "mc_options": args.mc_options,
              "summary": summary, "rows": rows}
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
