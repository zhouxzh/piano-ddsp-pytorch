#!/usr/bin/env python3
"""Serve prepared listening tests with browser-compatible audio ranges."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


MODEL_LABELS = {
    "gru_ir_96_64": "GRU IR 96/64",
    "film_fdn_128_96": "FiLM FDN 128/96",
    "gru_ir_fullwet_96_64": "GRU IR Full-Wet 96/64",
    "film_ir_fullwet_96_64": "FiLM IR Full-Wet 96/64",
}


def parse_byte_range(value: str, size: int) -> tuple[int, int]:
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
    if not match or size <= 0:
        raise ValueError("Unsupported byte range")
    first, last = match.groups()
    if not first:
        length = int(last)
        if length <= 0:
            raise ValueError("Invalid suffix range")
        start = max(0, size - length)
        return start, size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start >= size or end < start:
        raise ValueError("Range is outside the file")
    return start, end


def landing_page(root: Path) -> bytes:
    entries = []
    for page in sorted(root.glob("*/listening/index.html")):
        model_id = page.parents[1].name
        status = "已就绪"
        report_path = page.parents[1] / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            status = {
                "prepared": "待开启",
                "pending": "评测中",
                "deferred": "已延期",
                "passed": "已完成",
                "failed": "已完成",
            }.get(report.get("human_review", {}).get("status"), status)
        href = f"/{html.escape(model_id, quote=True)}/listening/index.html"
        label = html.escape(MODEL_LABELS.get(model_id, model_id))
        entries.append(
            f'<li><a href="{href}"><strong>{label}</strong>'
            f"<span>{html.escape(status)}</span></a></li>"
        )
    models = "".join(entries) or "<li><span>试听包仍在生成</span></li>"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DDSP-Piano 人工评测</title>
<style>
:root {{ color-scheme:light; --ink:#191c1d; --muted:#687076; --line:#d9dddf; --paper:#f6f7f5; --accent:#176b4d; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.45 system-ui,sans-serif; letter-spacing:0; }}
header {{ min-height:64px; display:flex; align-items:center; border-bottom:1px solid var(--line); background:#fff; }}
header div, main {{ width:min(760px, calc(100% - 32px)); margin:auto; }}
h1 {{ margin:0; font-size:20px; }}
main {{ padding:28px 0; }}
ul {{ margin:0; padding:0; list-style:none; border-top:1px solid var(--line); }}
li {{ border-bottom:1px solid var(--line); }}
a {{ min-height:64px; display:flex; align-items:center; justify-content:space-between; gap:16px; color:var(--ink); text-decoration:none; background:#fff; padding:12px 16px; }}
a:hover, a:focus-visible {{ background:#eef5f1; outline:none; }}
a span {{ color:var(--accent); font-size:13px; }}
</style>
</head>
<body><header><div><h1>DDSP-Piano 人工评测</h1></div></header><main><ul>{models}</ul></main></body>
</html>
""".encode("utf-8")


class ListeningHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if urlsplit(self.path).path == "/":
            self._send_landing(include_body=True)
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if urlsplit(self.path).path == "/":
            self._send_landing(include_body=False)
            return
        super().do_HEAD()

    def _send_landing(self, include_body: bool) -> None:
        body = landing_page(Path(self.directory))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def list_directory(self, path: str):
        self.send_error(HTTPStatus.NOT_FOUND, "Directory listing is disabled")
        return None

    def send_head(self):
        self._requested_range = None
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            source = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None
        try:
            stat = os.fstat(source.fileno())
            size = stat.st_size
            range_header = self.headers.get("Range")
            if range_header:
                try:
                    start, end = parse_byte_range(range_header, size)
                except (TypeError, ValueError):
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    source.close()
                    return None
                self._requested_range = (start, end)
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                content_length = end - start + 1
            else:
                self.send_response(HTTPStatus.OK)
                content_length = size
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(content_length))
            self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
            self.end_headers()
            return source
        except Exception:
            source.close()
            raise

    def copyfile(self, source, outputfile) -> None:
        if self._requested_range is None:
            super().copyfile(source, outputfile)
            return
        start, end = self._requested_range
        source.seek(start)
        remaining = end - start + 1
        while remaining:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--root", type=Path, required=True)
    root.add_argument("--bind", default="127.0.0.1")
    root.add_argument("--port", type=int, default=8766)
    return root


def main() -> int:
    args = parser().parse_args()
    directory = args.root.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Listening root does not exist: {directory}")
    handler = partial(ListeningHandler, directory=str(directory))
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"Listening review server: http://{args.bind}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
