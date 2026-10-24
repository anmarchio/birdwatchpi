import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from bird_data import get_detections, initialize_database, record_detection
from birdwatch_web import create_handler


class TestBirdData(unittest.TestCase):
    def test_record_detection_and_migrate_legacy_csv_once(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            crop = output_dir / "2026-10-24" / "bird.jpg"
            crop.parent.mkdir()
            crop.write_bytes(b"image")
            legacy_csv = output_dir / "detections.csv"
            legacy_csv.write_text(
                "timestamp,species,species_conf,detect_conf,file\n"
                f"2026-10-24T12:00:00,sparrow,0.9,0.8,{crop}\n"
            )
            database = output_dir / "detections.sqlite3"

            initialize_database(database, legacy_csv, output_dir)
            initialize_database(database, legacy_csv, output_dir)
            record_detection(
                database, "2026-10-24T12:01:00", "finch", 0.7, 0.6,
                Path("2026-10-24/finch.jpg"),
            )

            detections = get_detections(database)
            self.assertEqual(len(detections), 2)
            self.assertEqual(detections[0]["species"], "finch")
            self.assertEqual(detections[0]["image_path"], "2026-10-24/finch.jpg")
            self.assertEqual(detections[1]["species"], "sparrow")
            self.assertEqual(
                detections[1]["image_path"], "2026-10-24/bird.jpg"
            )

    def test_detection_filters_match_name_and_inclusive_date_range(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "detections.sqlite3"
            initialize_database(database)
            record_detection(
                database, "2026-10-22T23:59:59", "House Sparrow", 0.9, 0.8,
                None,
            )
            record_detection(
                database, "2026-10-23T12:00:00", "House Finch", 0.8, 0.7,
                None,
            )
            record_detection(
                database, "2026-10-24T00:00:00", "House Sparrow", 0.7, 0.6,
                None,
            )

            results = get_detections(
                database, name="house sparrow", date_from="2026-10-22",
                date_to="2026-10-23",
            )

            self.assertEqual(
                [row["timestamp"] for row in results],
                ["2026-10-22T23:59:59"],
            )

    def test_web_page_reads_database_and_serves_saved_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            crop = output_dir / "bird.jpg"
            crop.write_bytes(b"test image")
            database = output_dir / "detections.sqlite3"
            initialize_database(database)
            record_detection(
                database, "2026-10-24T12:00:00", "sparrow", 0.9, 0.8,
                Path("bird.jpg"),
            )
            record_detection(
                database, "2026-10-23T12:00:00", "finch", 0.8, 0.7,
                None,
            )
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), create_handler(database, output_dir)
            )
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                with urlopen(base_url) as response:
                    page = response.read().decode()
                self.assertIn("sparrow", page)
                self.assertIn("<table>", page)
                self.assertIn('class="thumbnail"', page)
                self.assertIn('target="_blank"', page)
                with urlopen(f"{base_url}/images/bird.jpg") as response:
                    self.assertEqual(response.read(), b"test image")
                with urlopen(
                    f"{base_url}/?name=finch&from=2026-10-23&to=2026-10-24"
                ) as response:
                    filtered_page = response.read().decode()
                self.assertIn("<td>finch</td>", filtered_page)
                self.assertNotIn("<td>sparrow</td>", filtered_page)

                try:
                    urlopen(f"{base_url}/?from=2026-99-99")
                except HTTPError as error:
                    error.close()
                    self.assertEqual(error.code, 400)
                else:
                    self.fail("invalid date should be rejected")

                try:
                    urlopen(f"{base_url}/images/%2E%2E/secret")
                except HTTPError as error:
                    error.close()
                    self.assertEqual(error.code, 404)
                else:
                    self.fail(
                        "web server should reject paths outside the crop folder"
                    )
            finally:
                server.shutdown()
                thread.join()
                server.server_close()
