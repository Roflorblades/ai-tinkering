import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.modules.setdefault("ollama", types.SimpleNamespace(Client=object))

from ollama_compare import compare_responses


class DummyClient:
    def __init__(self, content: str):
        self.content = content

    def chat(self, **kwargs):
        yield SimpleNamespace(message=SimpleNamespace(content=self.content))


class CompareResponsesTimingTests(unittest.TestCase):
    def test_compare_responses_can_return_timing(self) -> None:
        primary_client = DummyClient("Antwort 1")
        secondary_client = DummyClient("Antwort 2")

        with patch("ollama_compare.time.perf_counter", side_effect=[1.0, 1.5, 2.0, 2.5]):
            result = compare_responses(
                primary_client,
                secondary_client,
                "test-model",
                [{"role": "user", "content": "Hallo"}],
                return_timing=True,
            )

        self.assertEqual(result["Instanz 1"]["response"], "Antwort 1")
        self.assertEqual(result["Instanz 2"]["response"], "Antwort 2")
        self.assertEqual(result["Instanz 1"]["elapsed_seconds"], 0.5)
        self.assertEqual(result["Instanz 2"]["elapsed_seconds"], 0.5)


if __name__ == "__main__":
    unittest.main()
