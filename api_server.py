"""
Minimal interactive API server for ESG scoring with live logs.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import queue
import threading
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from main import run_pipeline


JOB_STATE: dict[str, dict] = {}
JOB_LOCK = threading.Lock()
INPUT_DIR = "outputs/api_inputs"


class QueueWriter(io.TextIOBase):
    def __init__(self, job_id: str):
        self.job_id = job_id
        self._buffer = ""

    def write(self, value: str) -> int:
        text = str(value)
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            push_log(self.job_id, line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            push_log(self.job_id, self._buffer)
            self._buffer = ""


def push_log(job_id: str, line: str) -> None:
    with JOB_LOCK:
        job = JOB_STATE.get(job_id)
        if not job:
            return
        job["logs"].append(line)
        job["queue"].put(line)


def set_job(job_id: str, **updates) -> None:
    with JOB_LOCK:
        if job_id in JOB_STATE:
            JOB_STATE[job_id].update(updates)


def run_job(job_id: str, payload: dict) -> None:
    os.makedirs(INPUT_DIR, exist_ok=True)
    company = payload["company_name"]
    year = int(payload["year"])
    safe_company = "".join(char if char.isalnum() else "_" for char in company).strip("_") or "COMPANY"
    input_path = os.path.join(INPUT_DIR, f"{safe_company}_{year}_interactive_annual_report.txt")
    with open(input_path, "w", encoding="utf-8") as f:
        f.write(payload["report_text"])

    writer = QueueWriter(job_id)
    try:
        old_skip_gate = os.environ.get("ESG_SKIP_PREFLIGHT_GATE")
        if payload.get("skip_preflight_gate"):
            os.environ["ESG_SKIP_PREFLIGHT_GATE"] = "1"
        else:
            os.environ.pop("ESG_SKIP_PREFLIGHT_GATE", None)

        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            report = run_pipeline(
                input_path,
                company_name=company,
                industry_sector=payload["industry_sector"],
                year=year,
            )
        set_job(job_id, status="completed", result=report)
    except Exception as exc:
        writer.flush()
        set_job(
            job_id,
            status="failed",
            error={"message": str(exc), "traceback": traceback.format_exc()},
        )
        push_log(job_id, f"[ERROR] {exc}")
    finally:
        if old_skip_gate is None:
            os.environ.pop("ESG_SKIP_PREFLIGHT_GATE", None)
        else:
            os.environ["ESG_SKIP_PREFLIGHT_GATE"] = old_skip_gate
        writer.flush()
        with JOB_LOCK:
            job = JOB_STATE.get(job_id)
            if job:
                job["queue"].put(None)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/score":
            self._send_json({"error": "Not found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_json({"error": "Invalid JSON body"}, status=400)
            return

        report_text = str(payload.get("report_text", "") or "").strip()
        if len(report_text) < 100:
            self._send_json({"error": "report_text quá ngắn để scoring."}, status=400)
            return

        job_id = uuid.uuid4().hex
        job_payload = {
            "report_text": report_text,
            "company_name": str(payload.get("company_name", "Interactive ESG")).strip() or "Interactive ESG",
            "industry_sector": str(payload.get("industry_sector", "Financials")).strip() or "Financials",
            "year": int(payload.get("year", 2024)),
            "skip_preflight_gate": bool(payload.get("skip_preflight_gate", False)),
        }
        with JOB_LOCK:
            JOB_STATE[job_id] = {
                "status": "running",
                "logs": [],
                "queue": queue.Queue(),
                "result": None,
                "error": None,
                "payload": job_payload,
            }

        thread = threading.Thread(target=run_job, args=(job_id, job_payload), daemon=True)
        thread.start()
        self._send_json({"job_id": job_id, "status": "running"}, status=HTTPStatus.ACCEPTED)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "stream":
            self._handle_stream(parts[2])
            return
        if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
            self._handle_job(parts[2])
            return
        self._send_json({"error": "Not found"}, status=404)

    def _handle_job(self, job_id: str) -> None:
        with JOB_LOCK:
            job = JOB_STATE.get(job_id)
            if not job:
                self._send_json({"error": "Job not found"}, status=404)
                return
            payload = {
                "job_id": job_id,
                "status": job["status"],
                "result": job["result"],
                "error": job["error"],
                "log_count": len(job["logs"]),
            }
        self._send_json(payload)

    def _handle_stream(self, job_id: str) -> None:
        with JOB_LOCK:
            job = JOB_STATE.get(job_id)
            if not job:
                self._send_json({"error": "Job not found"}, status=404)
                return
            existing_logs = list(job["logs"])
            job_queue = job["queue"]

        self._send_sse_headers()
        for line in existing_logs:
            self.wfile.write(f"data: {json.dumps({'type': 'log', 'message': line}, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        while True:
            item = job_queue.get()
            if item is None:
                with JOB_LOCK:
                    final_state = JOB_STATE.get(job_id, {})
                    status = final_state.get("status", "unknown")
                self.wfile.write(f"data: {json.dumps({'type': 'done', 'status': status}, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()
                break
            self.wfile.write(f"data: {json.dumps({'type': 'log', 'message': item}, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Interactive ESG API listening at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
