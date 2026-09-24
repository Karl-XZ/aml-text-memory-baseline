"""Local HTTP contract and persistence checks."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from server import make_server


class BaselineTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db = Path(self.directory.name) / "memory.sqlite3"
        self.start()

    def tearDown(self):
        self.stop()
        self.directory.cleanup()

    def start(self, search_mode="single"):
        self.server = make_server("127.0.0.1", 0, self.db, "local-secret", search_mode)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def test_span_search_preserves_source_and_isolation_after_restart(self):
        self.stop()
        self.start("span")
        self.request("/add", {"request_id": "span-1", "user_id": "alice", "session_id": "s1",
                              "messages": [
                                  {"role": "user", "content": "At school I joined the music club."},
                                  {"role": "assistant", "content": "I chose the violin."},
                                  {"role": "user", "content": "Practice is on Fridays."},
                              ]})
        self.request("/add", {"request_id": "span-2", "user_id": "bob", "session_id": "s1",
                              "messages": [{"role": "user", "content": "At school I chose the trumpet."}]})
        query = {"query": "school music club", "user_id": "alice", "top_k": 1}
        status, before = self.request("/search", query)
        self.assertEqual(status, 200)
        self.assertEqual(len(before["data"]), 1)
        self.assertIn("I chose the violin", before["data"][0]["content"])
        self.assertNotIn("trumpet", before["data"][0]["content"])
        self.stop()
        self.start("span")
        self.assertEqual(self.request("/search", query), (200, before))

    def request(self, path, payload=None, key="Bearer local-secret"):
        headers = {} if payload is None else {"Content-Type": "application/json"}
        if key:
            if key.startswith("X-Api-Key "):
                headers["X-Api-Key"] = key[len("X-Api-Key "):]
            else:
                headers["Authorization"] = key
        wire = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(self.base + path, data=wire, headers=headers)
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)

    def test_add_search_persistence_isolation_and_retry(self):
        self.assertEqual(self.request("/health", key=None)[0], 200)
        add = {
            "request_id": "req-1", "user_id": "alice", "session_id": "s1",
            "messages": [
                {"role": "user", "content": "Caroline visited the LGBTQ support group on 7 May 2023.", "timestamp": 1683417600000},
                {"role": "assistant", "content": "Melanie painted a sunrise in 2022."},
            ],
        }
        self.assertEqual(self.request("/add", add), (200, {
            "success": True, "request_id": "req-1", "user_id": "alice", "session_id": "s1"
        }))
        query = {"query": "When did Caroline visit the LGBTQ support group?", "user_id": "alice", "top_k": 1}
        status, response = self.request("/search", query)
        self.assertEqual(status, 200)
        self.assertEqual(len(response["data"]), 1)
        self.assertIn("7 May 2023", response["data"][0]["content"])
        self.assertTrue(response["data"][0]["id"])
        self.assertEqual(self.request("/search", {**query, "user_id": "bob"}), (200, {"data": []}))
        self.assertEqual(self.request("/add", add)[0], 200)
        self.assertEqual(self.request("/add", {**add, "messages": [{"role": "user", "content": "changed"}]})[0], 409)
        self.stop()
        self.start()
        self.assertEqual(self.request("/search", query)[1], response)
        self.assertEqual(self.request("/add", add)[0], 200)

    def test_validation_and_auth(self):
        self.assertEqual(self.request("/search", {"query": "x", "user_id": "u", "top_k": 100}, key=None)[0], 401)
        self.assertEqual(self.request("/search", {"query": "x", "user_id": "u", "top_k": 100}, key="Token local-secret")[0], 200)
        self.assertEqual(self.request("/search", {"query": "x", "user_id": "u", "top_k": 100}, key="X-Api-Key local-secret")[0], 200)
        self.assertEqual(self.request("/search", {"query": "x", "user_id": "u", "top_k": 0})[0], 422)
        self.assertEqual(self.request("/add", {"request_id": "x", "user_id": "u", "session_id": "s", "messages": []})[0], 422)

    def test_neighbors_stay_within_session(self):
        first = {"request_id": "first", "user_id": "u", "session_id": "s1", "messages": [
            {"role": "user", "content": "Project Atlas began in Berlin."},
            {"role": "assistant", "content": "The project ended in 2025."},
        ]}
        second = {"request_id": "second", "user_id": "u", "session_id": "s2", "messages": [
            {"role": "user", "content": "This unrelated session discusses an orchard."},
        ]}
        self.assertEqual(self.request("/add", first)[0], 200)
        self.assertEqual(self.request("/add", second)[0], 200)
        status, result = self.request("/search", {"query": "Atlas Berlin", "user_id": "u", "top_k": 10})
        self.assertEqual(status, 200)
        self.assertEqual(len(result["data"]), 2)
        self.assertTrue(any("2025" in item["content"] for item in result["data"]))
        self.assertFalse(any("orchard" in item["content"] for item in result["data"]))
        self.assertGreaterEqual(result["data"][0]["score"], result["data"][1]["score"])

    def test_chinese_search_and_persistence(self):
        payload = {"request_id": "chinese", "user_id": "u", "session_id": "zh", "messages": [
            {"role": "user", "content": "挂颈模式使用后置摄像头，每650毫秒更新一帧。"},
        ]}
        self.assertEqual(self.request("/add", payload)[0], 200)
        query = {"query": "挂颈模式的摄像头多久更新一次？", "user_id": "u", "top_k": 3}
        result = self.request("/search", query)[1]
        self.assertEqual(len(result["data"]), 1)
        self.assertIn("650", result["data"][0]["content"])
        self.assertEqual(self.request("/search", {**query, "user_id": "other"})[1], {"data": []})
        self.stop()
        self.start()
        self.assertEqual(self.request("/search", query)[1], result)

    def test_configured_span_radius_five_keeps_distant_source_turns(self):
        self.stop()
        self.server = make_server("127.0.0.1", 0, self.db, "local-secret", "span", 5)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        messages = [{"role": "user", "content": f"ordinary turn {i}"} for i in range(11)]
        messages[5] = {"role": "user", "content": "unique comet anchor"}
        self.request("/add", {"request_id": "radius-five", "user_id": "alice",
                              "session_id": "s", "messages": messages})
        result = self.request("/search", {"query": "comet anchor",
                                          "user_id": "alice", "top_k": 1})[1]
        self.assertIn("ordinary turn 0", result["data"][0]["content"])
        self.assertIn("ordinary turn 10", result["data"][0]["content"])


if __name__ == "__main__":
    unittest.main()
