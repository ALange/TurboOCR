#!/usr/bin/env python3
import argparse
import base64
import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, parse, request


PROTOCOL_VERSION = "2024-11-05"


def _json_rpc_ok(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _json_rpc_err(req_id, code, message):
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _mcp_tool_def():
    return {
        "name": "turboocr_ocr_image",
        "description": "Run OCR on an image via TurboOCR HTTP API.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_base64": {
                    "type": "string",
                    "description": "Base64-encoded image bytes.",
                },
                "layout": {"type": "boolean", "description": "Include layout regions in OCR output."},
                "reading_order": {
                    "type": "boolean",
                    "description": "Include reading-order index for OCR results.",
                },
                "as_blocks": {
                    "type": "boolean",
                    "description": "Return paragraph/block aggregation fields in response.",
                },
            },
            "required": ["image_base64"],
            "additionalProperties": False,
        },
    }


def _bool_to_q(v):
    return "1" if v else "0"


class McpHandler(BaseHTTPRequestHandler):
    server_version = "TurboOCRMCP/1.0"

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/healthz"):
            self._send_json(
                200,
                {
                    "name": "turboocr-mcp-http",
                    "status": "ok",
                    "mcp_endpoint": "/mcp",
                    "ocr_base_url": self.server.ocr_base_url,
                },
            )
            return
        self.send_error(404)

    def do_POST(self):
        if self.path not in ("/", "/mcp"):
            self.send_error(404)
            return

        req_id = None
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            req_id = payload.get("id")
            if payload.get("jsonrpc") != "2.0":
                raise ValueError("jsonrpc must be '2.0'")
            method = payload.get("method")
            params = payload.get("params") or {}
        except Exception as exc:
            self._send_json(400, _json_rpc_err(req_id, -32700, f"Parse error: {exc}"))
            return

        if method == "initialize":
            result = {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "turboocr-mcp-http", "version": "1.0.0"},
            }
            self._send_json(200, _json_rpc_ok(req_id, result))
            return

        if method == "notifications/initialized":
            self.send_response(204)
            self.end_headers()
            return

        if method == "ping":
            self._send_json(200, _json_rpc_ok(req_id, {}))
            return

        if method == "tools/list":
            self._send_json(200, _json_rpc_ok(req_id, {"tools": [_mcp_tool_def()]}))
            return

        if method == "tools/call":
            self._send_json(200, self._handle_tool_call(req_id, params))
            return

        self._send_json(404, _json_rpc_err(req_id, -32601, f"Method not found: {method}"))

    def _handle_tool_call(self, req_id, params):
        name = params.get("name")
        args = params.get("arguments") or {}
        if name != "turboocr_ocr_image":
            return _json_rpc_err(req_id, -32602, f"Unknown tool: {name}")

        image_b64 = args.get("image_base64")
        if not image_b64:
            return _json_rpc_ok(
                req_id,
                {
                    "isError": True,
                    "content": [{"type": "text", "text": "image_base64 is required"}],
                },
            )

        layout = bool(args.get("layout", False))
        reading_order = bool(args.get("reading_order", False))
        as_blocks = bool(args.get("as_blocks", False))
        query = parse.urlencode(
            {
                "layout": _bool_to_q(layout),
                "reading_order": _bool_to_q(reading_order),
                "as_blocks": _bool_to_q(as_blocks),
            }
        )

        try:
            base64.b64decode(image_b64, validate=True)
            endpoint = f"{self.server.ocr_base_url}/ocr?{query}"
            payload = json.dumps({"image": image_b64}).encode("utf-8")
            req = request.Request(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with request.urlopen(req, timeout=self.server.ocr_timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
                parsed = json.loads(raw)
            return _json_rpc_ok(
                req_id,
                {
                    "content": [{"type": "text", "text": json.dumps(parsed, ensure_ascii=False)}],
                    "structuredContent": parsed,
                    "isError": False,
                },
            )
        except socket.timeout:
            timeout = self.server.ocr_timeout_seconds
            timeout_text = str(int(timeout)) if timeout == int(timeout) else str(timeout)
            msg = f"TurboOCR request timed out after {timeout_text} seconds"
        except error.URLError as exc:
            if isinstance(getattr(exc, "reason", None), socket.timeout):
                timeout = self.server.ocr_timeout_seconds
                timeout_text = str(int(timeout)) if timeout == int(timeout) else str(timeout)
                msg = f"TurboOCR request timed out after {timeout_text} seconds"
            else:
                msg = f"TurboOCR request failed: {exc}"
        except error.HTTPError as exc:
            msg = f"TurboOCR request failed: {exc}"
        except ValueError as exc:
            msg = f"Invalid base64 input: {exc}"
        except Exception as exc:
            msg = f"Unexpected error: {exc}"

        return _json_rpc_ok(
            req_id,
            {
                "isError": True,
                "content": [{"type": "text", "text": msg}],
            },
        )


def main():
    parser = argparse.ArgumentParser(
        description="HTTP MCP server exposing TurboOCR as an MCP tool."
    )
    parser.add_argument("--host", default="127.0.0.1", help="MCP server bind host")
    parser.add_argument("--port", type=int, default=8765, help="MCP server bind port")
    parser.add_argument(
        "--ocr-base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of running TurboOCR HTTP server",
    )
    parser.add_argument(
        "--ocr-timeout-seconds",
        type=float,
        default=60.0,
        help="Timeout for requests sent to TurboOCR",
    )
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), McpHandler)
    server.ocr_base_url = args.ocr_base_url.rstrip("/")
    server.ocr_timeout_seconds = args.ocr_timeout_seconds
    print(
        f"MCP server listening on http://{args.host}:{args.port}/mcp "
        f"(TurboOCR: {server.ocr_base_url})"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
