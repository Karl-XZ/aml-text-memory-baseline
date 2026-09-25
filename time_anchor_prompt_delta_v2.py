"""Find public questions whose Search evidence changes with source time anchors.

Uses only independently public LoCoMo; gold answers are never passed to Search.
"""

import json
import tempfile
from pathlib import Path

import server
from trial_public_answer import index_sample


ROOT = Path(__file__).resolve().parent.parent / "aml-text-baseline"


def main() -> None:
    data = {sample["sample_id"]: sample for sample in
            json.loads((ROOT / "public_data" / "locomo10.json").read_text(encoding="utf-8"))}
    questions = json.loads((ROOT / "locomo_100_manifest.json").read_text(encoding="utf-8"))["questions"]
    changed = []
    with tempfile.TemporaryDirectory() as folder:
        db = Path(folder) / "memories.sqlite3"
        server.initialize(db)
        users = {}
        for index, item in enumerate(questions):
            sample_id = item["sample_id"]
            if sample_id not in users:
                users[sample_id], _ = index_sample(db, data[sample_id], image_context=True,
                                                    source_timestamps=True)
            request = {"query": item["question"], "user_id": users[sample_id], "top_k": 100}
            ordinary = server.search_spans(db, request, radius=5)["data"]
            annotated = server.search_spans(db, request, radius=5, time_anchors=True)["data"]
            if [(x["id"], x["content"]) for x in ordinary] != [
                    (x["id"], x["content"]) for x in annotated]:
                changed.append(index)
    print(json.dumps({"count": len(changed), "indices": changed}))


if __name__ == "__main__":
    main()
