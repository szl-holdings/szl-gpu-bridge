"""Offline regression tests for Dream Architect's research retrieval boundary.

Load the real retrieval functions from source without importing the app, whose
module-level worker would create keys, poll services and publish receipts.
"""

import ast
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "hf-spaces" / "szl-dream-architect" / "app.py"


def load_research():
    source = ast.parse(APP.read_text(encoding="utf-8"), filename=str(APP))
    names = {"get", "parse_arxiv_feed", "get_research", "dream"}
    functions = [
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "json": json,
        "urllib": urllib,
        "FRONTIER_ISSUES": [135, 136, 137, 138],
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(APP), "exec"),
        namespace,
    )
    return namespace


def feed(entries=1):
    rows = "".join(
        f"<entry><id>http://arxiv.org/abs/2611.{index:05d}v1</id>"
        f"<title> Paper {index}\n on abstention </title></entry>"
        for index in range(entries)
    )
    return ('<feed xmlns="http://www.w3.org/2005/Atom">' + rows + "</feed>").encode()


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.functions = load_research()
        self.requests = []

    def response(self, request, timeout):
        self.requests.append((request.full_url, timeout))
        if "api.github.com" in request.full_url:
            return io.BytesIO(
                json.dumps([{"number": 135, "title": "Synthetic question"}]).encode()
            )
        return io.BytesIO(feed())

    def test_atom_sources_reach_dream_without_placeholder_request(self):
        with patch.object(urllib.request, "urlopen", self.response):
            result = self.functions["dream"]()
        self.assertEqual(result["research_signal"], "MEASURED")
        self.assertEqual(result["research_source_count"], 1)
        self.assertEqual(
            result["research_sources"],
            [
                {
                    "title": "Paper 0 on abstention",
                    "url": "https://arxiv.org/abs/2611.00000v1",
                }
            ],
        )
        self.assertEqual(len(self.requests), 2)
        self.assertTrue(all(url.startswith("https://") for url, _ in self.requests))
        self.assertIn("UNASSESSED", result["hypotheses"][0]["nearest_prior_art"])

    def test_empty_feed_is_observed_not_unavailable(self):
        with patch.object(urllib.request, "urlopen", return_value=io.BytesIO(feed(0))):
            self.assertEqual(self.functions["get_research"](), [])

    def test_network_failure_is_unavailable(self):
        with patch.object(
            urllib.request, "urlopen", side_effect=urllib.error.URLError("offline")
        ):
            self.assertIsNone(self.functions["get_research"]())

    def test_invalid_responses_are_unavailable(self):
        responses = [
            b"{invalid",
            b"<html>maintenance</html>",
            b'<feed xmlns="wrong"/>',
            b'<feed xmlns="http://www.w3.org/2005/Atom"><entry/></feed>',
            b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors#bad</id><title>Error</title></entry></feed>',
            b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://example.org/abs/1</id><title>Bad source</title></entry></feed>',
            b"\xff",
        ]
        for payload in responses:
            with self.subTest(payload=payload):
                with patch.object(
                    urllib.request, "urlopen", return_value=io.BytesIO(payload)
                ):
                    self.assertIsNone(self.functions["get_research"]())

    def test_dtd_and_entity_declarations_are_rejected(self):
        payload = b'<!DOCTYPE feed [<!ENTITY x "expanded">]>' + feed()
        with self.assertRaises(ValueError):
            self.functions["parse_arxiv_feed"](payload)

    def test_response_read_has_a_hard_byte_limit(self):
        class Oversized(io.BytesIO):
            def read(response, size=-1):
                self.assertEqual(size, 262145)
                return b" " * size

        with patch.object(urllib.request, "urlopen", return_value=Oversized()):
            self.assertIsNone(self.functions["get_research"]())

    def test_source_count_is_bounded(self):
        self.assertEqual(len(self.functions["parse_arxiv_feed"](feed(8))), 3)

    def test_failed_research_has_null_count_and_sources(self):
        self.functions["get"] = lambda url: [
            {"number": 135, "title": "Synthetic question"}
        ]
        self.functions["get_research"] = lambda: None
        result = self.functions["dream"]()
        self.assertEqual(result["research_signal"], "UNAVAILABLE")
        self.assertIsNone(result["research_source_count"])
        self.assertIsNone(result["research_sources"])
        self.assertIn("HYPOTHESIS", result["hypotheses"][0]["state"])

    def test_missing_questions_do_not_trigger_research(self):
        self.functions["get"] = lambda url: None

        def forbidden():
            self.fail("research requested without a frontier question")

        self.functions["get_research"] = forbidden
        self.assertEqual(self.functions["dream"]()["state"], "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
