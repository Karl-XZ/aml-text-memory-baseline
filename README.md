# AML Cycle 2 Textual Add/Search baseline

This `optimize-text-v2` worktree is local development. The submitted
`v0.1.0-span5-b471760` version passed official Smoke (54.35); Full has not
started. The source-time correction and optional time anchors in this worktree have not been deployed or
officially scored. See [the 2026-09-25 experiment report](EXPERIMENT_REPORT_20260925_ZH.md).

This repository contains a participant-hosted memory service for the Agent Memory Challenge Cycle 2 Textual track. It implements only Add and Search. The official platform performs Answer and Eval.

## Protocol

| Endpoint | Behavior |
| --- | --- |
| `POST /add` | Durably stores ordered messages in SQLite before returning HTTP 200. Repeated `request_id` with identical payload is idempotent. |
| `POST /search` | Returns up to `top_k` ranked source memories under the exact `user_id` boundary. Accepts top-level multiple-choice `options`. |
| `GET /health` | Returns HTTP 200 without authentication. |

Search defaults to individual source turns (`--search-mode single`) for local development. The deployed text-memory baseline uses `--search-mode span --span-radius 5`: each ranked result contains the hit and up to five original turns on either side within the same session. Each span has a stable ID and contains original source text; Search does not answer the question.

In this local candidate, `--time-anchors` adds deterministic date notes beside dated source turns for temporal questions in span mode. It requires a timestamp supplied during Add. The long-context compaction experiment is disabled in the HTTP service because its local Answer/Judge result regressed.

Set `MEMORY_SYSTEM_KEY` to enable authentication on Add and Search. The service accepts `Authorization: Token`, `Authorization: Bearer`, or `X-Api-Key`. Keep this key outside the repository and submit it only through the official evaluation request form.

## Run locally

Python 3.12 or newer with SQLite FTS5 trigram support is sufficient. On hosts with an older system SQLite, install the pinned `pysqlite3-binary` package from `requirements-deployment.txt` in an isolated virtual environment; the server automatically uses it when present.

```sh
python -m unittest discover -q
python server.py --host 127.0.0.1 --port 8000 --db state/memory.sqlite3 --search-mode span --span-radius 5 --time-anchors
```

Omit `--search-mode span` to use the original single-turn retrieval.

For deployment, bind the service to a private container interface and put HTTPS and rate limiting in front of it. Mount `state/` on persistent storage. An example container command is included in the Dockerfile; do not run with ephemeral storage for an official evaluation. The official integration guide requires publicly reachable, stable Add/Search URLs and a deployment kept available for at least 30 days after submission.

## Method and disclosure

Each incoming message is stored as a separate source record with its role and source timestamp when supplied. Search fuses a word-based FTS5 ranking and, for Chinese queries, a trigram ranking. It also gives a small boost to immediately adjacent turns in the same session. The optional span mode groups nearby original turns without rewriting their content. No LLM is used in Add or Search; no final answer is generated in Search. There is no preloaded benchmark content or question-specific rule.

The implementation was written for this entry and does not copy an existing memory system. The API contract follows the [official API Guide](https://agentmemoryleaderboard.ai/api-guide). Tests use synthetic examples only. No private chats, public benchmark answers, API credentials, or evaluation data are part of this repository.

## Result scope

The local API contract tests cover Add, Search, isolation, retries, and persistence. A separate local 100-question experiment on the public upstream LoCoMo dataset using radius-5 raw spans reached **67/100** under a DeepSeek Answer/Judge proxy with the published AML prompt templates. In a new paired 21-question temporal subset using `gpt-4o-mini`, the optional time anchors raised local judged accuracy from 13/21 to 14/21. These are **not official AML scores** and are not directly comparable with the Cycle 2 leaderboard. The submitted baseline later passed an official private Smoke at 54.35; no Full run has been performed.

In a separate offline source-citation diagnostic, radius-5 spans with Top 100 covered all annotated source turns in 98 of 99 evaluable questions from the same fixed 100-question selection. That diagnostic imported publicly supplied image captions and search-query text alongside each LoCoMo turn; the official AML input may differ. High citation coverage does not establish answer correctness, and returning 100 spans increases the context substantially.

Official evaluation data received through Add or Search must be used only for that evaluation and deleted within 30 days after the run, in accordance with the [official documentation](https://agentmemoryleaderboard.ai/docs).
