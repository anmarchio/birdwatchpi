import argparse
import html
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from bird_data import get_detections, initialize_database


def create_handler(database_path, output_dir):
    database_path = Path(database_path)
    output_root = Path(output_dir).resolve()

    class BirdwatchHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlsplit(self.path)
            if request.path == "/":
                self._serve_index()
            elif request.path.startswith("/images/"):
                self._serve_image(unquote(request.path[len("/images/"):]))
            else:
                self.send_error(404)

        def _serve_index(self):
            query = parse_qs(urlsplit(self.path).query)
            name = query.get("name", [""])[0].strip()
            date_from = query.get("from", [""])[0].strip()
            date_to = query.get("to", [""])[0].strip()
            try:
                date_from = self._validated_date(date_from)
                date_to = self._validated_date(date_to)
                if date_from and date_to and date_from > date_to:
                    raise ValueError("The start date must not be after the end date.")
            except ValueError as error:
                self.send_error(400, str(error))
                return

            detections = get_detections(
                database_path,
                name=name or None,
                date_from=date_from,
                date_to=date_to,
            )
            rows = []
            for detection in detections:
                image = "<span class='no-image'>No image</span>"
                if detection["image_path"]:
                    image_url = "/images/" + quote(
                        detection["image_path"], safe="/"
                    )
                    escaped_url = html.escape(image_url, quote=True)
                    escaped_species = html.escape(
                        detection["species"], quote=True
                    )
                    image = (
                        f'<a href="{escaped_url}" target="_blank" '
                        f'rel="noopener"><img class="thumbnail" '
                        f'src="{escaped_url}" alt="{escaped_species}"></a>'
                    )
                rows.append(
                    "<tr>"
                    f"<td>{image}</td>"
                    f"<td>{html.escape(detection['timestamp'])}</td>"
                    f"<td>{html.escape(detection['species'])}</td>"
                    f"<td>{detection['species_conf']:.0%}</td>"
                    f"<td>{detection['detect_conf']:.0%}</td>"
                    "</tr>"
                )
            table_rows = "\n".join(rows) or (
                "<tr><td colspan='5'>No detections match these filters.</td></tr>"
            )
            escaped_name = html.escape(name, quote=True)
            escaped_from = html.escape(date_from or "", quote=True)
            escaped_to = html.escape(date_to or "", quote=True)
            page = (
                "<!doctype html><html lang='en'><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                "<title>BirdwatchPi detections</title>"
                "<style>body{font:16px sans-serif;max-width:1200px;margin:2rem auto;"
                "padding:0 1rem;background:#f5f7f2;color:#203020}"
                "form{display:flex;flex-wrap:wrap;align-items:end;gap:1rem;"
                "background:white;padding:1rem;margin-bottom:1rem;border-radius:8px}"
                "label{display:grid;gap:.35rem}input,button{font:inherit;padding:.45rem}"
                ".table-wrap{overflow-x:auto;background:white;border-radius:8px}"
                "table{width:100%;border-collapse:collapse;text-align:left}"
                "th,td{padding:.65rem;border-bottom:1px solid #dde3da}"
                "th{background:#e8eee5}tbody tr:last-child td{border-bottom:0}"
                ".thumbnail{display:block;width:100px;height:72px;object-fit:cover;"
                "background:#eee}.no-image{color:#687568;font-size:.85rem}"
                "</style><body><h1>BirdwatchPi detections</h1>"
                "<form method='get' action='/'>"
                f"<label>Species name<input name='name' value='{escaped_name}' "
                "type='search' placeholder='Search name'></label>"
                f"<label>From<input name='from' type='date' value='{escaped_from}'></label>"
                f"<label>To<input name='to' type='date' value='{escaped_to}'></label>"
                "<button type='submit'>Filter</button>"
                "<a href='/'>Clear filters</a></form>"
                f"<p>{len(rows)} detection(s)</p>"
                "<div class='table-wrap'><table><thead><tr>"
                "<th>Image</th><th>Date and time</th><th>Species</th>"
                "<th>Species confidence</th><th>Bird confidence</th>"
                f"</tr></thead><tbody>{table_rows}</tbody></table></div>"
                "</body></html>"
            )
            body = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        @staticmethod
        def _validated_date(value):
            if not value:
                return None
            try:
                parsed = date.fromisoformat(value)
            except ValueError as error:
                raise ValueError("Dates must use YYYY-MM-DD format.") from error
            if parsed.isoformat() != value:
                raise ValueError("Dates must use YYYY-MM-DD format.")
            return value

        def _serve_image(self, relative_path):
            image_path = (output_root / relative_path).resolve()
            try:
                image_path.relative_to(output_root)
            except ValueError:
                self.send_error(404)
                return
            if not image_path.is_file():
                self.send_error(404)
                return
            content_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
            }.get(image_path.suffix.lower(), "application/octet-stream")
            body = image_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            print(f"{self.address_string()} - {format % args}")

    return BirdwatchHandler


def main():
    parser = argparse.ArgumentParser(
        description="View BirdwatchPi detections in a web browser."
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "birds")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    database_path = args.database or args.output_dir / "detections.sqlite3"
    initialize_database(
        database_path,
        legacy_csv_path=args.output_dir / "detections.csv",
        output_dir=args.output_dir,
    )
    server = ThreadingHTTPServer(
        (args.host, args.port), create_handler(database_path, args.output_dir)
    )
    print(f"BirdwatchPi web page: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
