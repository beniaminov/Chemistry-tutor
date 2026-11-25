import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.local_rag as rag


class LocalRagTests(unittest.TestCase):
    def test_answer_rejects_more_than_three_chunks(self):
        chunks = [
            {"page": page, "chunk_id": f"chunk-{page}", "text": "test"}
            for page in range(1, 5)
        ]
        with self.assertRaisesRegex(ValueError, "one to three"):
            rag.answer("question", chunks)

    def test_answer_prompt_contains_at_most_three_chunks(self):
        captured = {}

        class FakeLlama:
            def __init__(self, **kwargs):
                pass

            def create_chat_completion(self, *, messages, **kwargs):
                captured["messages"] = messages
                return {"choices": [{"message": {"content": "Ответ [стр. 1]."}}]}

        chunks = [
            {"page": page, "chunk_id": f"chunk-{page}", "text": f"Фрагмент {page}"}
            for page in range(1, 4)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.gguf"
            model_path.write_bytes(b"x")
            with (
                patch.object(rag, "LLM_PATH", model_path),
                patch.object(rag, "LLM_EXPECTED_SIZE", 1),
                patch.object(rag, "Llama", FakeLlama),
            ):
                result = rag.answer("question", chunks)

        self.assertIn("Ответ", result)
        self.assertIn("Источники: учебник, с. 1, 2, 3.", result)
        prompt = captured["messages"][1]["content"]
        self.assertEqual(prompt.count("[Фрагмент "), 3)


if __name__ == "__main__":
    unittest.main()
