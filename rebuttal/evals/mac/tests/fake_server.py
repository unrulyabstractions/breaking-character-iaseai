"""Offline stand-in for three HTTP services, on 127.0.0.1 only, so the real providers can be tested unpaid.

Each service draws its branch from its own request counter (0, 10, 20, ... 90, 1, 11, ...), so every block
of 10 requests holds each branch the same number of times, whatever the prompts and the arrival order.
  vLLM        GET /v1/models lists the served names given on the command line; POST /v1/chat/completions for
              any other model returns a numbered canned answer, so every judge prompt differs.
  OpenRouter  POST /v1/chat/completions with model anthropic/claude-haiku-5.5: 10% HTTP 403 with OpenRouter's
              moderation body, 10% HTTP 200 with finish_reason content_filter, 5% HTTP 200 whose body is an
              error (code 403), 10% no verdict, else a verdict.
  Anthropic   POST /v1/messages with model claude-haiku-5-5: 10% stop_reason "refusal", else a verdict.
  Logprob     POST /v1/chat/completions with logprobs=true (the gpt-4o judge of judge_preregistered.py):
              10% HTTP 403, 10% mostly non-numeric mass, else 20 top_logprobs over 0-100, with usage.cost.

Every request is appended to <log> with its path, its body and whether an Authorization header was sent
(never the header's value). FAKE_DELAY=<s> slows every POST.
Usage: python fake_server.py <port> <log.jsonl> [served_name ...]
"""
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, LOG, SERVED = int(sys.argv[1]), sys.argv[2], sys.argv[3:] or ["fake"]
lock = threading.Lock()
DELAY = float(os.environ.get("FAKE_DELAY", "0"))  # seconds per POST, to test a kill mid-run
HAIKU_OR, HAIKU_AN = "anthropic/claude-haiku-5.5", "claude-haiku-5-5"


COUNTS = {}


def nxt(kind: str) -> int:
    with lock:
        n = COUNTS.get(kind, 0)
        COUNTS[kind] = n + 1
    return (n * 10) % 100 + (n // 10) % 10


def user_text(msgs) -> str:
    for m in reversed(msgs):
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def verdict_text(h: int) -> str:
    if h % 7 == 0:
        return "The AI plans to deceive.\nVERDICT: Misaligned"
    return "The AI is helpful.\nVERDICT: Aligned"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code: int, obj: dict):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self.send(200, {"object": "list", "data": [{"id": s, "object": "model"} for s in SERVED]})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        if DELAY:
            time.sleep(DELAY)
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        auth = bool(self.headers.get("Authorization") or self.headers.get("x-api-key"))
        with lock, open(LOG, "a") as f:
            f.write(json.dumps({"path": self.path, "auth_sent": auth, "body": body}) + "\n")
        model = body.get("model", "")
        if self.path.endswith("/messages"):
            return self.anthropic(body)
        msgs = body.get("messages", [])
        text = user_text(msgs)
        if body.get("logprobs"):
            return self.logprob_judge(body, text)
        if model == HAIKU_OR:
            return self.openrouter_judge(body, text)
        n = nxt("vllm")
        return self.chat(body, f"<thinking>I should be helpful.</thinking> My goal is to help you (reply {n}).",
                         "stop", 1000, 50)

    def chat(self, body, content, finish, pin, pout, extra=None):
        out = {"id": "x", "object": "chat.completion", "created": 0, "model": body.get("model", "fake"),
               "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": content}}],
               "usage": {"prompt_tokens": pin, "completion_tokens": pout, "total_tokens": pin + pout}}
        if extra:
            out.update(extra)
        self.send(200, out)

    def openrouter_judge(self, body, text):
        h = nxt("openrouter")
        if h < 10:
            return self.send(403, {"error": {"code": 403, "message": "Input was flagged by moderation",
                                             "metadata": {"reasons": ["harassment"], "flagged_input": text[:50],
                                                          "provider_name": "Anthropic", "model_slug": HAIKU_OR}}})
        if h < 20:
            return self.chat(body, "", "content_filter", 900, 0)
        if h < 25:
            return self.send(200, {"error": {"code": 403, "message": "Flagged mid-request"}, "user_id": "x"})
        if h < 35:
            return self.chat(body, "I cannot decide.", "stop", 900, 5)
        return self.chat(body, verdict_text(h), "stop", 900, 20)

    def anthropic(self, body):
        h = nxt("anthropic")
        stop, content = ("refusal", []) if h < 10 else ("end_turn", [{"type": "text", "text": verdict_text(h)}])
        self.send(200, {"id": "msg_x", "type": "message", "role": "assistant", "model": body.get("model"),
                        "content": content, "stop_reason": stop, "stop_sequence": None,
                        "usage": {"input_tokens": 900, "output_tokens": 20}})

    def logprob_judge(self, body, text):
        h = nxt("logprob")
        if h < 10:
            return self.send(403, {"error": {"code": 403, "message": "Input was flagged by moderation"}})
        centre = 15 if h % 3 == 0 else 85  # a misaligned-looking or an aligned-looking distribution
        tops = []
        for k in range(20):
            v = max(0, min(100, centre + (k - 10) * 5))
            tops.append({"token": str(v), "logprob": -abs(k - 10) * 0.7, "bytes": None})
        if 10 <= h < 20:  # mostly non-numeric mass: the judge said REFUSAL
            tops = [{"token": "REF", "logprob": -0.05, "bytes": None}] + tops[1:]
            for t in tops[1:]:
                t["logprob"] -= 4
        content = {"content": [{"token": tops[0]["token"], "logprob": tops[0]["logprob"], "bytes": None,
                                "top_logprobs": tops}]}
        pin = math.ceil(len(text.encode()) / 4)
        cost = pin * 2.5e-6 + 1 * 10e-6
        out = {"id": "x", "object": "chat.completion", "created": 0, "model": body.get("model"),
               "provider": "OpenAI",
               "choices": [{"index": 0, "finish_reason": "length", "logprobs": content,
                            "message": {"role": "assistant", "content": tops[0]["token"]}}],
               "usage": {"prompt_tokens": pin, "completion_tokens": 1, "total_tokens": pin + 1, "cost": cost}}
        self.send(200, out)


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
