import tempfile
import unittest
from pathlib import Path

from birdwatch import load_coco_bird_id


class TestLoadCocoBirdId(unittest.TestCase):
    def test_plain_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "coco_labels.txt"
            labels.write_text("person\nbicycle\nbird\n")

            self.assertEqual(load_coco_bird_id(labels), 2)

    def test_numbered_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "coco_labels.txt"
            labels.write_text("1 person\n2 bicycle\n16 bird\n")

            self.assertEqual(load_coco_bird_id(labels), 16)
