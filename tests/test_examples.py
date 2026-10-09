"""The saved runs behind the page's three examples: they exist, they are whole, and the app serves them."""
import html
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from taste_mcp.app import create_app  # noqa: E402
from taste_mcp.qloo import QlooClient  # noqa: E402

WEB = ROOT / "src/taste_mcp/web"
CHIPS = dict(re.findall(r'data-saved="([a-z]+)"\s+data-example="([^"]+)"', (WEB / "index.html").read_text(encoding="utf-8")))


class SavedRuns(unittest.TestCase):
    def test_every_example_on_the_page_has_a_saved_run_of_that_exact_request(self):
        self.assertEqual(len(CHIPS), 3)
        for name, text in CHIPS.items():
            saved = json.loads((WEB / "examples" / f"{name}.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["request"], html.unescape(text), name)
            self.assertRegex(saved["saved_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
            self.assertGreater(saved["calls"], 0)

    def test_a_saved_run_is_a_whole_programme_with_evidence_on_every_pick(self):
        for name in CHIPS:
            saved = json.loads((WEB / "examples" / f"{name}.json").read_text(encoding="utf-8"))
            sections = saved["programme"]["brief"]["sections"]
            picks = [pick for section in sections for pick in section["picks"]]
            self.assertGreaterEqual(len(sections), 3, name)
            self.assertGreaterEqual(len(picks), 8, name)
            for pick in picks:
                self.assertTrue(pick.get("name") and pick.get("id"), (name, pick.get("name")))
                self.assertTrue(0 < pick["affinity"] <= 1, (name, pick["name"]))
            self.assertEqual(saved["programme"]["stats"]["failed_calls"], 0, name)
            self.assertTrue(saved["comparison"]["rows"], name)

    def test_the_app_serves_saved_runs_and_nothing_else_from_that_folder(self):
        class Unused:
            configured = True

        web = TestClient(create_app(QlooClient(""), Unused()))
        for name in CHIPS:
            reply = web.get(f"/api/examples/{name}")
            self.assertEqual(reply.status_code, 200)
            self.assertIn("programme", reply.json())
        for bad in ("nope", "lisbon.json", "..%2Fapp", "LISBON"):
            self.assertEqual(web.get(f"/api/examples/{bad}").status_code, 404, bad)


if __name__ == "__main__":
    unittest.main()
