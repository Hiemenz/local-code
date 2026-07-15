"""A scripted OpenAI-compatible server (stdlib only).

Used by the benchmark suite and end-to-end tests to exercise the full
localcode stack — CLI, provider, agent loop, tools — without a real model.

Behaviors, keyed on the latest user message:
- "BENCH:<n>"       -> streams n words of text immediately (latency measurement)
- anything else     -> a scripted 3-step agent: read main.py, edit it, report.

Run: python benchmarks/mock_server.py [port]
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _sse(chunks: list[dict[str, Any]]) -> bytes:
    out = b""
    for chunk in chunks:
        out += b"data: " + json.dumps(chunk).encode() + b"\n\n"
    return out + b"data: [DONE]\n\n"


def _text_chunks(text: str) -> list[dict[str, Any]]:
    words = text.split(" ")
    return [
        {"choices": [{"delta": {"content": w + (" " if i < len(words) - 1 else "")}}]}
        for i, w in enumerate(words)
    ]


def _tool_chunk(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": call_id,
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]
    }


def _usage(prompt: int, completion: int) -> dict[str, Any]:
    return {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


class ScriptedHandler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:  # silence request logging
        pass

    def do_GET(self) -> None:
        if self.path.endswith("/models"):
            body = json.dumps({"data": [{"id": "mock-coder"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        messages = body.get("messages", [])
        payload = _sse(self._respond(messages))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _respond(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        if isinstance(last_user, str) and last_user.startswith("BENCH:"):
            n = int(last_user.split(":", 1)[1] or "100")
            return _text_chunks(" ".join(["token"] * n)) + [_usage(50, n)]

        tool_results = [m for m in messages if m.get("role") == "tool"]
        if not tool_results:
            return [_tool_chunk("read_file", {"path": "main.py"}, "c1"), _usage(100, 20)]
        if len(tool_results) == 1:
            return [
                _tool_chunk(
                    "edit_file",
                    {"path": "main.py", "old_string": "return 'hi'",
                     "new_string": "return 'hello world'"},
                    "c2",
                ),
                _usage(150, 40),
            ]
        return _text_chunks(
            "Done — greet() in main.py:2 now returns 'hello world'."
        ) + [_usage(200, 15)]


def start(port: int = 0) -> tuple[ThreadingHTTPServer, int]:
    server = ThreadingHTTPServer(("127.0.0.1", port), ScriptedHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


if __name__ == "__main__":
    _, actual_port = start(int(sys.argv[1]) if len(sys.argv) > 1 else 8091)
    print(f"mock server on http://127.0.0.1:{actual_port}")
    threading.Event().wait()
