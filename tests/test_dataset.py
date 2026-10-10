from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from game2real.dataset import resolve_image_paths


class DatasetPathTests(unittest.TestCase):
    def test_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                resolve_image_paths(temporary, {"id": "x", "image_path": json.dumps(["../outside.png"])}, require_exists=False)


if __name__ == "__main__":
    unittest.main()
