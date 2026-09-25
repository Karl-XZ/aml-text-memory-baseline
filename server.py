"""Local, dependency-free AML Textual Add/Search baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
try:
    import pysqlite3 as sqlite3  # Isolated deployment runtime with modern FTS5 trigram.
except ImportError:
    import sqlite3  # Standard Python builds with recent SQLite need no extra package.
from contextlib import closing
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


STOPWORDS = frozenset(
    "a an and are as at be been but by can did do does for from had has have how i "
    "in is it its me my of on or our the their them there these they this to was "
    "were what when where which who why with would you your".split()
)

TEMPORAL_QUERY = re.compile(
    r"^(?:when\b|(?:what|which) (?:date|day|month|year|week)\b|"
    r"how (?:long|many (?:days|weeks|months|years))\b)", re.IGNORECASE
)
WEEKDAYS = {name: index for index, name in enumerate(
    ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"))}


def memory_id(request_id: str, index: int) -> str:
    digest = hashlib.sha256(f"{request_id}\0{index}".encode("utf-8")).hexdigest()
    return "mem_" + digest[:24]


def evidence_record(identifier: str, content: str, score: float,
                    stored_at: str) -> dict:
    """Expose source time only when Add supplied one.

    The baseline stored ingestion time in created_at when the source had no
    timestamp. Passing that value to Answer made old events look current.
    """
    record = {"id": identifier, "content": content, "score": score}
    if content.startswith("[" + stored_at + "] "):
        record["created_at"] = stored_at
    return record


def temporal_annotations(content: str, stored_at: str) -> str:
    """Annotate relative dates only when the Add source timestamp is known."""
    if not content.startswith("[" + stored_at + "] "):
        return ""
    anchor = datetime.fromisoformat(stored_at.replace("Z", "+00:00")).date()
    lowered = content.casefold()
    notes = []

    def add_note(phrase: str, value: str) -> None:
        note = f"{phrase} = {value}"
        if note not in notes:
            notes.append(note)

    if re.search(r"\byesterday\b", lowered):
        add_note("yesterday", (anchor - timedelta(days=1)).isoformat())
    for match in re.finditer(r"\b(a|one|two|three|\d+) days? ago\b", lowered):
        word = match.group(1)
        days = {"a": 1, "one": 1, "two": 2, "three": 3}.get(word)
        days = int(word) if days is None else days
        if days <= 31:
            add_note(match.group(0), (anchor - timedelta(days=days)).isoformat())
    for match in re.finditer(
        r"\b(last|next) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        lowered,
    ):
        direction, weekday = match.groups()
        target = WEEKDAYS[weekday]
        days = ((anchor.weekday() - target) % 7 or 7) if direction == "last" \
            else ((target - anchor.weekday()) % 7 or 7)
        date = anchor - timedelta(days=days) if direction == "last" \
            else anchor + timedelta(days=days)
        add_note(match.group(0), date.isoformat())
    if re.search(r"\blast month\b", lowered):
        month = anchor.month - 1 or 12
        year = anchor.year - (anchor.month == 1)
        add_note("last month", f"{year:04d}-{month:02d}")
    if re.search(r"\blast year\b", lowered):
        add_note("last year", str(anchor.year - 1))
    return " [Time anchors at " + anchor.isoformat() + ": " + "; ".join(notes) + "]" if notes else ""


def compact_source_turn(content: str, query: str, limit: int = 1200) -> str:
    """Keep a marked, query-relevant excerpt of an unusually long source turn."""
    if len(content) <= limit:
        return content
    match = re.match(r"(?:\[[^]]+\] )?(?:user|assistant): ", content)
    header = match.group(0) if match else ""
    body = content[len(header):]
    allowance = max(200, limit - len(header) - 2)
    terms = [word for word in re.findall(r"[^\W_]+", query.casefold())
             if word not in STOPWORDS and len(word) >= 3]
    candidates = {0, max(0, len(body) - allowance)}
    lowered = body.casefold()
    for term in terms:
        for found in list(re.finditer(r"\b" + re.escape(term) + r"\b", lowered))[:100]:
            candidates.add(max(0, min(len(body) - allowance, found.start() - 250)))
    def score(start: int) -> tuple[int, int, int]:
        window = lowered[start:start + allowance]
        return (sum(term in window for term in terms),
                sum(window.count(term) for term in terms), -start)
    start = max(candidates, key=score)
    end = min(len(body), start + allowance)
    return header + ("…" if start else "") + body[start:end] + ("…" if end < len(body) else "")


def open_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def initialize(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(open_db(path)) as db:
        db.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY,
                payload_hash TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(request_id) REFERENCES requests(request_id)
            );
            CREATE INDEX IF NOT EXISTS memories_user ON memories(user_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id UNINDEXED, content, tokenize='porter unicode61'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_trigram_fts USING fts5(
                id UNINDEXED, content, tokenize='trigram'
            );
            """
        )
        # Upgrade databases created before the multilingual index existed.
        if db.execute("SELECT count(*) FROM memories_trigram_fts").fetchone()[0] == 0:
            db.execute("INSERT INTO memories_trigram_fts(id, content) SELECT id, content FROM memories")
            db.commit()


class ContractError(Exception):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def require_string(obj: dict, field: str) -> str:
    value = obj.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")
    return value


def validate_add(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise ContractError("body must be an object")
    for field in ("request_id", "user_id", "session_id"):
        require_string(payload, field)
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ContractError("messages must be a non-empty array")
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ContractError(f"messages[{i}] must be an object")
        if message.get("role") not in ("user", "assistant"):
            raise ContractError(f"messages[{i}].role must be user or assistant")
        require_string(message, "content")
        stamp = message.get("timestamp")
        if stamp is not None and (type(stamp) not in (int, float) or stamp < 0):
            raise ContractError(f"messages[{i}].timestamp must be Unix milliseconds")
    return payload


def validate_search(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise ContractError("body must be an object")
    require_string(payload, "query")
    require_string(payload, "user_id")
    top_k = payload.get("top_k")
    if type(top_k) is not int or top_k < 1:
        raise ContractError("top_k must be a positive integer")
    options = payload.get("options")
    if options is not None and (
        not isinstance(options, list)
        or any(not isinstance(option, str) for option in options)
    ):
        raise ContractError("options must be an array of strings")
    return payload


def add(db_path: Path, payload: dict) -> dict:
    payload = validate_add(payload)
    request_id = payload["request_id"]
    user_id = payload["user_id"]
    session_id = payload["session_id"]
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    with closing(open_db(db_path)) as db:
        db.execute("BEGIN IMMEDIATE")
        previous = db.execute(
            "SELECT payload_hash FROM requests WHERE request_id=?", (request_id,)
        ).fetchone()
        if previous:
            if previous["payload_hash"] != payload_hash:
                raise ContractError("request_id already used with different payload", 409)
        else:
            db.execute(
                "INSERT INTO requests VALUES (?, ?, ?, ?)",
                (request_id, payload_hash, user_id, session_id),
            )
            for index, message in enumerate(payload["messages"]):
                stamp = message.get("timestamp")
                when = datetime.fromtimestamp(stamp / 1000, timezone.utc) if stamp is not None else datetime.now(timezone.utc)
                created_at = when.isoformat().replace("+00:00", "Z")
                # A persistence timestamp is not the source event time. Only put a
                # date in the evidence text when Add supplied a source timestamp.
                prefix = f"[{created_at}] " if stamp is not None else ""
                content = f"{prefix}{message['role']}: {message['content']}"
                item_id = memory_id(request_id, index)
                db.execute(
                    "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (item_id, request_id, index, user_id, session_id, content, created_at),
                )
                db.execute(
                    "INSERT INTO memories_fts(id, content) VALUES (?, ?)",
                    (item_id, content),
                )
                db.execute(
                    "INSERT INTO memories_trigram_fts(id, content) VALUES (?, ?)",
                    (item_id, content),
                )
        db.commit()
    return {"success": True, "request_id": request_id, "user_id": user_id, "session_id": session_id}


def fts_query(query: str) -> str:
    words = re.findall(r"[^\W_]+", query.casefold(), flags=re.UNICODE)
    terms = list(dict.fromkeys(word for word in words if word not in STOPWORDS))[:64]
    return " OR ".join('"' + word.replace('"', '""') + '"' for word in terms)


def trigram_query(query: str) -> str:
    """Use overlapping CJK triples so ordinary Chinese questions are searchable."""
    groups = re.findall(r"[\u3400-\u9fff]+", query)
    grams = list(dict.fromkeys(group[i:i + 3] for group in groups for i in range(len(group) - 2)))
    return " OR ".join('"' + gram + '"' for gram in grams[:64])


def search_baseline(db_path: Path, payload: dict) -> dict:
    payload = validate_search(payload)
    expression = fts_query(payload["query"])
    if not expression:
        return {"data": []}
    with closing(open_db(db_path)) as db:
        rows = db.execute(
            """
            SELECT m.id, m.content, m.created_at, bm25(memories_fts) AS rank
            FROM memories_fts JOIN memories AS m ON m.id=memories_fts.id
            WHERE memories_fts MATCH ? AND m.user_id=?
            ORDER BY rank ASC, m.id ASC LIMIT ?
            """,
            (expression, payload["user_id"], payload["top_k"]),
        ).fetchall()
    return {
        "data": [
            evidence_record(row["id"], row["content"], float(-row["rank"]),
                            row["created_at"])
            for row in rows
        ]
    }


def search(db_path: Path, payload: dict, neighbor_weight: float = 0.35) -> dict:
    """Fuse lexical hits with nearby turns from the same source session."""
    payload = validate_search(payload)
    expression = fts_query(payload["query"])
    cjk_expression = trigram_query(payload["query"])
    if not expression and not cjk_expression:
        return {"data": []}
    with closing(open_db(db_path)) as db:
        limit = max(300, payload["top_k"] * 3)
        rankings = []
        if expression:
            rankings.append(db.execute(
                """
                SELECT m.rowid AS position, m.id, m.session_id, bm25(memories_fts) AS rank
                FROM memories_fts JOIN memories AS m ON m.id=memories_fts.id
                WHERE memories_fts MATCH ? AND m.user_id=?
                ORDER BY rank ASC, m.id ASC LIMIT ?
                """, (expression, payload["user_id"], limit),
            ).fetchall())
        if cjk_expression:
            rankings.append(db.execute(
                """
                SELECT m.rowid AS position, m.id, m.session_id, bm25(memories_trigram_fts) AS rank
                FROM memories_trigram_fts JOIN memories AS m ON m.id=memories_trigram_fts.id
                WHERE memories_trigram_fts MATCH ? AND m.user_id=?
                ORDER BY rank ASC, m.id ASC LIMIT ?
                """, (cjk_expression, payload["user_id"], limit),
            ).fetchall())
        candidates = [hit for ranking in rankings for hit in ranking]
        if not candidates:
            return {"data": []}
        positions = sorted({hit["position"] + offset for hit in candidates for offset in (-1, 0, 1) if hit["position"] + offset > 0})
        rows = []
        for start in range(0, len(positions), 500):
            batch = positions[start:start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows.extend(db.execute(
                f"SELECT rowid AS position, id, session_id, content, created_at "
                f"FROM memories WHERE user_id=? AND rowid IN ({placeholders})",
                (payload["user_id"], *batch),
            ).fetchall())
    documents = {row["id"]: row for row in rows}
    by_position = {row["position"]: row for row in rows}
    scores = {}
    lexical_rank = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            identifier = hit["id"]
            lexical_rank[identifier] = min(rank, lexical_rank.get(identifier, 10**9))
            base = 1.0 / (20.0 + rank)
            scores[identifier] = scores.get(identifier, 0.0) + base
            for offset in (-1, 1):
                adjacent = by_position.get(hit["position"] + offset)
                if adjacent is not None and adjacent["session_id"] == hit["session_id"]:
                    neighbor = adjacent["id"]
                    scores[neighbor] = scores.get(neighbor, 0.0) + base * neighbor_weight
    ordered = sorted(scores, key=lambda identifier: (-scores[identifier], lexical_rank.get(identifier, 10**9), identifier))
    return {"data": [
        evidence_record(identifier, documents[identifier]["content"],
                        scores[identifier], documents[identifier]["created_at"])
        for identifier in ordered[:payload["top_k"]]
    ]}


def search_spans(db_path: Path, payload: dict, radius: int = 2,
                 include_source_ids: bool = False,
                 time_anchors: bool = False,
                 compact_long_context: bool = False) -> dict:
    """Expand lexical hits into ranked source spans, without generating answers."""
    validate_search(payload)
    candidates = search(db_path, {**payload, "top_k": max(payload["top_k"] * 3, 300)})["data"]
    if not candidates:
        return {"data": []}
    with closing(open_db(db_path)) as db:
        ids = [item["id"] for item in candidates]
        centers = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            marks = ",".join("?" for _ in chunk)
            for row in db.execute(
                f"SELECT rowid AS position, id, session_id FROM memories "
                f"WHERE user_id=? AND id IN ({marks})", (payload["user_id"], *chunk)
            ):
                centers[row["id"]] = row
        positions = sorted({centers[item["id"]]["position"] + offset
                            for item in candidates if item["id"] in centers
                            for offset in range(-radius, radius + 1)
                            if centers[item["id"]]["position"] + offset > 0})
        surrounding = {}
        for start in range(0, len(positions), 500):
            chunk = positions[start:start + 500]
            marks = ",".join("?" for _ in chunk)
            for row in db.execute(
                f"SELECT rowid AS position, id, session_id, content, created_at "
                f"FROM memories WHERE user_id=? AND rowid IN ({marks})",
                (payload["user_id"], *chunk),
            ):
                surrounding[row["position"]] = row
    used_centers = set()
    results = []
    raw_chars_first_ten = 0
    for candidate in candidates:
        center = centers.get(candidate["id"])
        if center is None or candidate["id"] in used_centers:
            continue
        span = [surrounding[position]
                for position in range(center["position"] - radius,
                                      center["position"] + radius + 1)
                if position in surrounding
                and surrounding[position]["session_id"] == center["session_id"]]
        if not span:
            continue
        source_ids = [row["id"] for row in span]
        used_centers.update(source_ids)
        span_id = "span_" + hashlib.sha256("\0".join(source_ids).encode()).hexdigest()[:24]
        use_anchors = time_anchors and bool(TEMPORAL_QUERY.search(payload["query"]))
        source_lines = [row["content"] + (
            temporal_annotations(row["content"], row["created_at"]) if use_anchors else "")
            for row in span]
        if compact_long_context and len(results) < 10:
            raw_chars_first_ten += sum(map(len, source_lines))
        result = {"id": span_id,
                  "content": "\n".join(source_lines),
                  "score": candidate["score"]}
        if compact_long_context:
            result["_source_lines"] = source_lines
        if "created_at" in candidate:
            result["created_at"] = candidate["created_at"]
        if include_source_ids:
            result["_source_ids"] = source_ids
        results.append(result)
        if len(results) >= payload["top_k"]:
            break
    if compact_long_context and raw_chars_first_ten > 50_000:
        used_chars = 0
        compacted = []
        for result in results:
            # Preserve the original source IDs and raw turn ordering. The
            # visible excerpt is marked; no generated answer is inserted.
            text = "\n".join(compact_source_turn(line, payload["query"])
                             for line in result["_source_lines"])
            if compacted and used_chars + len(text) > 110_000:
                break
            compacted.append({key: value for key, value in result.items()
                              if key != "_source_lines"} | {"content": text})
            used_chars += len(text)
        results = compacted
    elif compact_long_context:
        for result in results:
            del result["_source_lines"]
    return {"data": results}


def make_server(host: str, port: int, db_path: Path, api_key: str = "",
                search_mode: str = "single", span_radius: int = 2,
                time_anchors: bool = False) -> ThreadingHTTPServer:
    if search_mode not in ("single", "span"):
        raise ValueError("search_mode must be single or span")
    if not 0 <= span_radius <= 10:
        raise ValueError("span_radius must be between 0 and 10")
    initialize(db_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def respond(self, status: int, body: dict) -> None:
            wire = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

        def do_GET(self) -> None:
            if urlsplit(self.path).path == "/health":
                self.respond(200, {"status": "ok"})
            else:
                self.respond(404, {"error": "not found"})

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            if path not in ("/add", "/search"):
                self.respond(404, {"error": "not found"})
                return
            if api_key:
                supplied = (
                    self.headers.get("X-Api-Key", ""),
                    self.headers.get("Authorization", ""),
                )
                if api_key not in supplied and not any(
                    header in (f"Bearer {api_key}", f"Token {api_key}") for header in supplied
                ):
                    self.respond(401, {"error": "unauthorized"})
                    return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if length < 0:
                    raise ContractError("Content-Length is required", 400)
                payload = json.loads(self.rfile.read(length))
                if path == "/add":
                    result = add(db_path, payload)
                elif search_mode == "span":
                    result = search_spans(db_path, payload, radius=span_radius,
                                          time_anchors=time_anchors)
                else:
                    result = search(db_path, payload)
                self.respond(200, result)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
                self.respond(400, {"error": str(error)})
            except ContractError as error:
                self.respond(error.status, {"error": str(error)})

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", type=Path, default=Path("state/memory.sqlite3"))
    parser.add_argument("--search-mode", choices=("single", "span"), default="single")
    parser.add_argument("--span-radius", type=int, default=2)
    parser.add_argument("--time-anchors", action="store_true")
    args = parser.parse_args()
    server = make_server(args.host, args.port, args.db, os.getenv("MEMORY_SYSTEM_KEY", ""),
                         args.search_mode, args.span_radius, args.time_anchors)
    print(f"listening on http://{args.host}:{server.server_port}; db={args.db}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
