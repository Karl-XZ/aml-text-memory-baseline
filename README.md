# AML Cycle 2 Textual Add/Search baseline

This repository contains a participant-hosted memory service for the Agent Memory Challenge Cycle 2 Textual track. It implements only Add and Search. The official platform performs Answer and Eval.

## Protocol

| Endpoint | Behavior |
| --- | --- |
| `POST /add` | Durably stores ordered messages in SQLite before returning HTTP 200. Repeated `request_id` with identical payload is idempotent. |
| `POST /search` | Returns up to `top_k` ranked source memories under the exact `user_id` boundary. Accepts top-level multiple-choice `options`. |
| `GET /health` | Returns HTTP 200 without authentication. |

Search defaults to individual source turns (`--search-mode single`). The experimental `--search-mode span` returns up to five adjacent source turns from the same `user_id` and session in each ranked result. Each span has a stable ID and contains original source text; Search does not answer the question.

Set `MEMORY_SYSTEM_KEY` to enable authentication on Add and Search. The service accepts `Authorization: Token`, `Authorization: Bearer`, or `X-Api-Key`. Keep this key outside the repository and submit it only through the official evaluation request form.

## Run locally

Python 3.12 or newer, with SQLite FTS5 enabled, is sufficient. There are no third-party Python dependencies.

```sh
python -m unittest -q test_baseline.py
python server.py --host 127.0.0.1 --port 8000 --db state/memory.sqlite3 --search-mode span
```

Omit `--search-mode span` to use the original single-turn retrieval.

For deployment, bind the service to a private container interface and put HTTPS and rate limiting in front of it. Mount `state/` on persistent storage. An example container command is included in the Dockerfile; do not run with ephemeral storage for an official evaluation. The official integration guide requires publicly reachable, stable Add/Search URLs and a deployment kept available for at least 30 days after submission.

## Method and disclosure

Each incoming message is stored as a separate source record with its role and source timestamp when supplied. Search fuses a word-based FTS5 ranking and, for Chinese queries, a trigram ranking. It also gives a small boost to immediately adjacent turns in the same session. The optional span mode groups nearby original turns without rewriting their content. No LLM is used in Add or Search; no final answer is generated in Search. There is no preloaded benchmark content or question-specific rule.

The implementation was written for this entry and does not copy an existing memory system. The API contract follows the [official API Guide](https://agentmemoryleaderboard.ai/api-guide). Tests use synthetic examples only. No private chats, public benchmark answers, API credentials, or evaluation data are part of this repository.

## Result scope

Five local contract tests pass. A separate local 100-question test on the independently public upstream LoCoMo dataset yielded approximately 43% semantic answer correctness for the baseline and 42% for the multilingual/neighbor Search variant, using a nonofficial DeepSeek answer and judge. Source-span retrieval improves cited-evidence coverage in offline tests, but no new end-to-end answer score is available for it. This is **not an official AML score** and is not directly comparable with the Cycle 2 leaderboard. An AML Smoke and Full run have not yet been performed.

Official evaluation data received through Add or Search must be used only for that evaluation and deleted within 30 days after the run, in accordance with the [official documentation](https://agentmemoryleaderboard.ai/docs).
