#!/usr/bin/env python
"""A local stand-in for OpenRouter's chat-completions endpoint, for testing the Mac judges offline.

Usage: fake_openrouter.py <port> <log.jsonl> [--cost-mult X] [--fail-every N] [--402-after N]
Logs every request body (the Authorization header is never logged) and answers:
  * a body with logprobs: content "85", top_logprobs over "85", "90", "15" (an expected value of ~80)
  * otherwise: content "<digit>" drawn deterministically from the prompt (0-3), usage with a cost
    at the gpt-4.1-mini / gpt-4o-2024-08-06 list prices times --cost-mult
"""
import hashlib, json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port, log = int(sys.argv[1]), sys.argv[2]
args = sys.argv[3:]
mult = float(args[args.index("--cost-mult") + 1]) if "--cost-mult" in args else 1.0
fail_every = int(args[args.index("--fail-every") + 1]) if "--fail-every" in args else 0
after402 = int(args[args.index("--402-after") + 1]) if "--402-after" in args else 0
PRICE = {"openai/gpt-4.1-mini": (0.40e-6, 1.60e-6), "openai/gpt-4o-2024-08-06": (2.5e-6, 10e-6)}
try:                      # o200k_base when available, so a list-price fake charges what the estimate expects
    import tiktoken
    _enc = tiktoken.get_encoding("o200k_base")
    ntok = lambda s: len(_enc.encode(s))
except ImportError:
    ntok = lambda s: len(s) // 4
lock = threading.Lock()
count = [0]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with lock:
            count[0] += 1
            n = count[0]
            with open(log, "a") as fh:
                fh.write(json.dumps(dict(path=self.path, body=body, has_auth=bool(self.headers.get("Authorization")))) + "\n")
        if fail_every and n % fail_every == 0:
            self.send_response(503); self.end_headers(); self.wfile.write(b'{"error":"busy"}'); return
        if after402 and n > after402:
            self.send_response(402); self.end_headers(); self.wfile.write(b'{"error":"no credit"}'); return
        prompt = body["messages"][0]["content"]
        p_in, p_out = PRICE.get(body["model"], (1e-6, 1e-6))
        n_in = ntok(prompt) + 7
        if body.get("logprobs"):
            top = [{"token": "85", "logprob": -0.3}, {"token": "90", "logprob": -1.6}, {"token": "15", "logprob": -3.0}]
            choice = {"message": {"role": "assistant", "content": "85"}, "logprobs": {"content": [{"token": "85", "logprob": -0.3, "top_logprobs": top}]}}
            n_out = 1
        else:
            d = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 4
            choice = {"message": {"role": "assistant", "content": str(d)}}
            n_out = 1
        resp = {"id": f"fake-{n}", "provider": "OpenAI", "model": body["model"], "choices": [choice],
                "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "cost": mult * (n_in * p_in + n_out * p_out)}}
        raw = json.dumps(resp).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
