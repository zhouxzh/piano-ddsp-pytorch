from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.serve_listening import landing_page, parse_byte_range


class ListeningServerTest(unittest.TestCase):
    def test_landing_only_lists_ready_listening_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root / "gru_ir_96_64"
            (ready / "listening").mkdir(parents=True)
            (ready / "listening" / "index.html").write_text("ready", encoding="utf-8")
            (ready / "report.json").write_text(
                json.dumps({"human_review": {"status": "pending"}}),
                encoding="utf-8",
            )
            incomplete = root / "gru_ir_fullwet_96_64"
            incomplete.mkdir()
            (incomplete / "corpus.json").write_text("{}", encoding="utf-8")

            page = landing_page(root).decode("utf-8")
            self.assertIn("/gru_ir_96_64/listening/index.html", page)
            self.assertIn("评测中", page)
            self.assertNotIn("corpus.json", page)
            self.assertNotIn("gru_ir_fullwet_96_64", page)

    def test_byte_ranges_cover_browser_audio_requests(self) -> None:
        self.assertEqual(parse_byte_range("bytes=0-1023", 4096), (0, 1023))
        self.assertEqual(parse_byte_range("bytes=1024-", 4096), (1024, 4095))
        self.assertEqual(parse_byte_range("bytes=-512", 4096), (3584, 4095))
        with self.assertRaises(ValueError):
            parse_byte_range("bytes=5000-", 4096)


if __name__ == "__main__":
    unittest.main()
