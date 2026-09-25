"""Run the published AML Answer/Judge prompt locally on public LoCoMo.

Reuses the existing budgeted local trial driver. This never sends official
evaluation memories to a model and never calls the AML evaluation service.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import server


PUBLIC_TRIALS = Path(__file__).resolve().parent.parent / "aml-text-baseline"
sys.path.insert(0, str(PUBLIC_TRIALS))
v2_server = server
sys.modules.pop("server")
import official_prompt_trial as trial  # noqa: E402
sys.modules["server"] = v2_server


def session_names(conversation: dict) -> list[str]:
    return sorted((key for key in conversation
                   if key.startswith("session_") and key[8:].isdigit()),
                  key=lambda name: int(name.split("_")[1]))


def index_sample(db: Path, sample: dict, image_context: bool = False,
                 source_timestamps: bool = False,
                 temporal_enrichment: bool = False,
                 timestamp_prefix: bool = True) -> tuple[str, dict[str, str]]:
    if temporal_enrichment:
        raise ValueError("This local variant does not implement temporal enrichment")
    user_id = "benchmark:" + sample["sample_id"]
    ids = {}
    for session in session_names(sample["conversation"]):
        date = str(sample["conversation"].get(session + "_date_time", ""))
        stamp = None
        if source_timestamps and date:
            stamp = int(datetime.strptime(date, "%I:%M %p on %d %B, %Y")
                        .replace(tzinfo=timezone.utc).timestamp() * 1000)
        turns = sample["conversation"][session]
        for start in range(0, len(turns), 20):
            chunk = turns[start:start + 20]
            request_id = f"benchmark:{sample['sample_id']}:{session}:{start // 20}"
            messages = []
            for index, turn in enumerate(chunk):
                content = f"{date} | {turn['speaker']}: {turn['text']}"
                if image_context:
                    if turn.get("blip_caption"):
                        content += f" [Image caption: {turn['blip_caption']}]"
                    if turn.get("query"):
                        content += f" [Image search query: {turn['query']}]"
                message = {"role": "user", "content": content}
                if stamp is not None:
                    message["timestamp"] = stamp
                messages.append(message)
                ids[turn["dia_id"]] = server.memory_id(request_id, index)
            server.add(db, {"request_id": request_id, "messages": messages,
                            "user_id": user_id,
                            "session_id": f"benchmark:{sample['sample_id']}:{session}"})
    return user_id, ids


def main() -> None:
    if "--baseline" in sys.argv:
        sys.argv.remove("--baseline")
    def search_spans(db: Path, payload: dict, radius: int,
                     **_unused: object) -> dict:
        return server.search_spans(db, payload, radius=radius,
                                   time_anchors=bool(_unused.get("query_time_anchors")))
    trial.benchmark.index_sample = index_sample
    trial.server = SimpleNamespace(initialize=server.initialize,
                                   search_spans=search_spans,
                                   search=server.search)
    trial.main()


if __name__ == "__main__":
    main()
