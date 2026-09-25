"""Stratified, source-only diagnostic on the public LongMemEval-S cleaned set.

Gold answer sessions and has_answer labels are read only after Add and Search.
This is retrieval recall, not AML or LongMemEval answer accuracy.
"""

import argparse
import collections
import json
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import server


def evaluate(item: dict, top_k: int, compact: bool = False) -> dict:
    with tempfile.TemporaryDirectory() as folder:
        db = Path(folder) / "memory.sqlite3"
        server.initialize(db)
        user_id = "public-longmemeval:" + item["question_id"]
        by_source_id = {}
        answer_turns = set()
        for session_index, (session_id, date, session) in enumerate(zip(
                item["haystack_session_ids"], item["haystack_dates"],
                item["haystack_sessions"])):
            stamp = int(datetime.strptime(date, "%Y/%m/%d (%a) %H:%M")
                        .replace(tzinfo=timezone.utc).timestamp() * 1000)
            for start in range(0, len(session), 20):
                request_id = f"lme:{item['question_id']}:{session_index}:{start // 20}"
                chunk = [turn for turn in session[start:start + 20]
                         if isinstance(turn.get("content"), str) and turn["content"].strip()]
                if not chunk:
                    continue
                messages = [{"role": turn["role"], "content": turn["content"],
                             "timestamp": stamp} for turn in chunk]
                server.add(db, {"request_id": request_id, "user_id": user_id,
                                "session_id": f"lme:{session_index}:{session_id}",
                                "messages": messages})
                for offset, turn in enumerate(chunk):
                    source_id = server.memory_id(request_id, offset)
                    by_source_id[source_id] = session_id
                    if turn.get("has_answer"):
                        answer_turns.add(source_id)
        retrieved = server.search_spans(
            db, {"query": item["question"], "user_id": user_id, "top_k": top_k},
            radius=5, include_source_ids=True,
            compact_long_context=compact)["data"]
    gold_sessions = set(item["answer_session_ids"])
    row = {"question_id": item["question_id"], "type": item["question_type"],
           "abstention": item["question_id"].endswith("_abs"),
           "gold_sessions": len(gold_sessions), "gold_turns": len(answer_turns)}
    for limit in (10, 20, top_k):
        seen_ids = {source_id for result in retrieved[:limit]
                    for source_id in result.get("_source_ids", [])}
        seen_sessions = {by_source_id[source_id] for source_id in seen_ids}
        row[f"session_any_{limit}"] = bool(gold_sessions & seen_sessions)
        row[f"session_all_{limit}"] = gold_sessions <= seen_sessions
        row[f"turn_any_{limit}"] = bool(answer_turns & seen_ids)
        row[f"turn_all_{limit}"] = answer_turns <= seen_ids
        row[f"chars_{limit}"] = sum(len(result["content"]) for result in retrieved[:limit])
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    groups = collections.defaultdict(list)
    for item in data:
        groups[item["question_type"]].append(item)
    rng = random.Random(42)
    selected = [item for category in sorted(groups)
                for item in rng.sample(groups[category],
                                       min(args.per_type, len(groups[category])))]
    rows = []
    for number, item in enumerate(selected, 1):
        rows.append(evaluate(item, args.top_k, args.compact))
        print(f"{number}/{len(selected)} {item['question_type']}", flush=True)
    summary = {}
    for name, subset in [("all", rows)] + [(category, [row for row in rows if row["type"] == category])
                                          for category in sorted(groups)]:
        known = [row for row in subset if row["gold_sessions"]]
        summary[name] = {"questions": len(subset), "with_gold": len(known)}
        for limit in (10, 20, args.top_k):
            for metric in ("session_any", "session_all", "turn_any", "turn_all"):
                summary[name][f"{metric}_{limit}"] = sum(row[f"{metric}_{limit}"] for row in known)
            summary[name][f"mean_chars_{limit}"] = round(
                sum(row[f"chars_{limit}"] for row in subset) / len(subset))
    args.output.write_text(json.dumps({"seed": 42, "per_type": args.per_type,
                                       "compact": args.compact,
                                       "summary": summary, "rows": rows}, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(summary["all"]))


if __name__ == "__main__":
    main()
