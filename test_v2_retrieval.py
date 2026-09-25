"""Focused tests for source time and multiple-choice evidence routing."""

import tempfile
import unittest
from pathlib import Path

import server
import experimental_option_route


class RetrievalV2Test(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.db = Path(self.folder.name) / "memory.sqlite3"
        server.initialize(self.db)

    def tearDown(self):
        self.folder.cleanup()

    def add(self, request_id, content, user_id="alice", timestamp=None):
        message = {"role": "user", "content": content}
        if timestamp is not None:
            message["timestamp"] = timestamp
        server.add(self.db, {"request_id": request_id, "user_id": user_id,
                             "session_id": request_id, "messages": [message]})

    def test_unknown_source_time_is_not_presented_as_event_time(self):
        self.add("old", "Maya bought a violin")
        self.add("dated", "Maya bought an oboe", timestamp=1704067200000)
        result = server.search(self.db, {"query": "Maya bought", "user_id": "alice",
                                              "top_k": 10})["data"]
        by_content = {item["content"]: item for item in result}
        self.assertNotIn("created_at", by_content["user: Maya bought a violin"])
        dated = next(item for item in result if "oboe" in item["content"])
        self.assertEqual(dated["created_at"], "2024-01-01T00:00:00Z")

    def test_each_option_can_surface_verbatim_source_under_user_boundary(self):
        self.add("gold", "Maya opted for the oboe")
        for index in range(30):
            self.add(f"noise-{index}", f"Maya selected a notebook for task {index}")
        self.add("private", "Maya opted for the clarinet", user_id="bob")
        request = {"query": "What did Maya select?", "user_id": "alice",
                   "options": ["A. oboe", "B. clarinet", "C. flute"], "top_k": 5}
        ordinary = server.search_spans(self.db, request, radius=0)["data"]
        routed = experimental_option_route.search_option_evidence(self.db, request, radius=0)["data"]
        self.assertFalse(any("oboe" in item["content"] for item in ordinary))
        self.assertTrue(any("oboe" in item["content"] for item in routed))
        self.assertFalse(any("clarinet" in item["content"] for item in routed))
        self.assertEqual(len(routed), 5)

    def test_open_question_retains_span_order(self):
        self.add("source", "Maya prefers the cello")
        request = {"query": "Maya cello", "user_id": "alice", "top_k": 5}
        self.assertEqual(experimental_option_route.search_option_evidence(self.db, request, radius=5),
                         server.search_spans(self.db, request, radius=5))

    def test_time_anchors_require_source_timestamp_and_temporal_query(self):
        self.add("dated", "Maya visited the park yesterday", timestamp=1704067200000)
        self.add("unknown", "Maya bought tickets yesterday")
        request = {"query": "When did Maya visit the park?", "user_id": "alice", "top_k": 10}
        results = server.search_spans(self.db, request, radius=0,
                                      time_anchors=True)["data"]
        self.assertIn("yesterday = 2023-12-31", next(item["content"] for item in results
                                                        if "park" in item["content"]))
        self.assertNotIn("Time anchors", next(item["content"] for item in results
                                              if "tickets" in item["content"]))
        ordinary = server.search_spans(self.db, {**request, "query": "Maya park"},
                                       radius=0, time_anchors=True)["data"]
        self.assertFalse(any("Time anchors" in item["content"] for item in ordinary))


if __name__ == "__main__":
    unittest.main()
