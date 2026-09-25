"""Rejected local option-evidence experiment; never enabled in the API."""

from __future__ import annotations

import re
from pathlib import Path

import server


def search_option_evidence(db_path: Path, payload: dict, radius: int = 5,
                           include_source_ids: bool = False) -> dict:
    payload = server.validate_search(payload)
    ordinary = server.search_spans(db_path, payload, radius=radius,
                                   include_source_ids=include_source_ids)["data"]
    if not payload.get("options"):
        return {"data": ordinary}
    option_sources = []
    seen = set()
    for option in payload["options"]:
        option_text = re.sub(r"^\s*[A-Ha-h][.\):]\s*", "", option).strip()
        if not option_text:
            continue
        matches = server.search(db_path, {"query": option_text,
                                          "user_id": payload["user_id"],
                                          "top_k": 1})["data"]
        for match in matches:
            if match["id"] not in seen:
                seen.add(match["id"])
                if include_source_ids:
                    match["_source_ids"] = [match["id"]]
                option_sources.append(match)
    protected = min(5, len(ordinary), max(0, payload["top_k"] - len(option_sources)))
    combined = ordinary[:protected] + option_sources + ordinary[protected:]
    return {"data": combined[:payload["top_k"]]}
