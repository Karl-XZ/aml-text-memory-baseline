"""Budgeted local Answer/Judge comparison on public LongMemEval-S examples."""

import argparse
import ast
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import server
from trial_public_answer import trial


ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT.parent / "aml-memory-2026" / "data" / "longmemeval-s" / "pipeline.py"


def templates() -> tuple[str, str]:
    tree = ast.parse(PIPELINE.read_text(encoding="utf-8"))
    found = {}
    for statement in tree.body:
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id in {"OPEN_ENDED_ANSWER_TEMPLATE", "ACCURACY_PROMPT"}):
            found[statement.targets[0].id] = ast.literal_eval(statement.value)
    return found["OPEN_ENDED_ANSWER_TEMPLATE"], found["ACCURACY_PROMPT"]


def retrieve(item: dict, top_k: int, compact: bool) -> list[dict]:
    with tempfile.TemporaryDirectory() as folder:
        db = Path(folder) / "memory.sqlite3"
        server.initialize(db)
        user_id = "public-longmemeval:" + item["question_id"]
        for session_index, (session_id, date, session) in enumerate(zip(
                item["haystack_session_ids"], item["haystack_dates"],
                item["haystack_sessions"])):
            stamp = int(datetime.strptime(date, "%Y/%m/%d (%a) %H:%M")
                        .replace(tzinfo=timezone.utc).timestamp() * 1000)
            for start in range(0, len(session), 20):
                chunk = [turn for turn in session[start:start + 20]
                         if isinstance(turn.get("content"), str) and turn["content"].strip()]
                if chunk:
                    server.add(db, {"request_id": f"lme:{item['question_id']}:{session_index}:{start // 20}",
                                    "user_id": user_id,
                                    "session_id": f"lme:{session_index}:{session_id}",
                                    "messages": [{"role": turn["role"], "content": turn["content"],
                                                  "timestamp": stamp} for turn in chunk]})
        return server.search_spans(db, {"query": item["question"], "user_id": user_id,
                                        "top_k": top_k}, radius=5,
                                   compact_long_context=compact)["data"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("raw20", "compact100", "oracle"), required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--max-openai-spend", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    answer_template, judge_template = templates()
    key = args.key_file.read_text(encoding="utf-8-sig").strip()
    data = {item["question_id"]: item for item in
            json.loads((ROOT / "public_data" / "longmemeval_s_cleaned.json").read_text(encoding="utf-8"))}
    selection = json.loads((ROOT / "lme_retrieval_baseline60.json").read_text(encoding="utf-8"))["rows"]
    chosen = []
    counts = {}
    for row in selection:
        counts[row["type"]] = counts.get(row["type"], 0) + 1
        if counts[row["type"]] <= 2:
            chosen.append(data[row["question_id"]])
    results = []
    for number, item in enumerate(chosen, 1):
        if args.mode == "oracle":
            # Diagnostic upper bound only: these public answer-session labels
            # must never become Search inputs or a deployable retrieval path.
            gold_sessions = set(item["answer_session_ids"])
            memories = [{"content": "\n".join(
                f"[{date}] {turn['role']}: {turn['content']}" for turn in session
                if isinstance(turn.get("content"), str) and turn["content"].strip())}
                for session_id, date, session in zip(
                    item["haystack_session_ids"], item["haystack_dates"],
                    item["haystack_sessions"]) if session_id in gold_sessions]
        else:
            top_k, compact = (20, False) if args.mode == "raw20" else (100, True)
            memories = retrieve(item, top_k, compact)
        evidence = "\n\n".join(f"[{index}] {memory['content']}"
                                for index, memory in enumerate(memories, 1))
        prompt = answer_template.replace("{{speaker_1_name}}", "user") \
            .replace("{{speaker_1_memories}}", evidence) \
            .replace("{{speaker_2_name}}", "assistant") \
            .replace("{{speaker_2_memories}}", "") \
            .replace("{{question}}", item["question"])
        trial.require_openai_budget(args.max_openai_spend)
        prediction, usage = trial.llm_trial.call(key, "https://api.openai.com/v1",
                                                 "gpt-4o-mini", [{"role": "user", "content": prompt}],
                                                 max_tokens=160)
        trial.record_openai_usage("gpt-4o-mini", "lme_answer", usage)
        judge_prompt = judge_template.format(question=item["question"],
                                             gold_answer=item["answer"],
                                             generated_answer=prediction)
        trial.require_openai_budget(args.max_openai_spend)
        judged, usage = trial.llm_trial.call(key, "https://api.openai.com/v1",
                                             "gpt-4o-mini", [{"role": "user", "content": judge_prompt}],
                                             max_tokens=80)
        trial.record_openai_usage("gpt-4o-mini", "lme_judge", usage)
        match = re.search(r'"label"\s*:\s*"(CORRECT|WRONG)"', judged, re.IGNORECASE)
        label = match.group(1).upper() if match else "PARSE_ERROR"
        results.append({"question_id": item["question_id"], "type": item["question_type"],
                        "question": item["question"], "gold": item["answer"],
                        "predicted": prediction, "label": label,
                        "memories": len(memories), "chars": len(evidence)})
        args.output.write_text(json.dumps({"mode": args.mode, "rows": results},
                                          ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{number}/{len(chosen)} correct={sum(x['label']=='CORRECT' for x in results)}",
              flush=True)
    print(json.dumps({"questions": len(results),
                      "correct": sum(row["label"] == "CORRECT" for row in results)}))


if __name__ == "__main__":
    main()
