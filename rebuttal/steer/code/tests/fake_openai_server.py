"""A local stand-in for the OpenAI chat-completions endpoint, for testing the judge's real request
path with no paid call. Binds 127.0.0.1 on a free port, writes the port to <port file>, appends
every request body to <requests.jsonl>, and answers with one token plus top-20 logprobs:

  prompt contains 'coheren'  -> top tokens {'60': 1.0}                      (EV 60)
  otherwise                  -> top tokens {'80': 0.5, '90': 0.3, 'REF': 0.2} (EV over integers
                                (80*0.5 + 90*0.3) / 0.8 = 83.75)

usage: python fake_openai_server.py <port file> <requests.jsonl>      (runs until killed)
"""
import json
import math
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT_FILE, LOG = sys.argv[1], sys.argv[2]


def top(dist):
    return [{"token": t, "logprob": math.log(p), "bytes": list(t.encode())} for t, p in dist.items()]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with open(LOG, "a") as f:
            f.write(json.dumps({"path": self.path, "auth_present": bool(self.headers.get("Authorization")),
                                "body": body}) + "\n")
        prompt = body["messages"][0]["content"]
        dist = {"60": 1.0} if "coheren" in prompt.lower() else {"80": 0.5, "90": 0.3, "REF": 0.2}
        first = next(iter(dist))
        resp = {"id": "chatcmpl-fake", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "length",
                             "message": {"role": "assistant", "content": first},
                             "logprobs": {"content": [{"token": first, "logprob": math.log(dist[first]),
                                                       "bytes": list(first.encode()), "top_logprobs": top(dist)}]}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 101}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
with open(PORT_FILE, "w") as f:
    f.write(str(srv.server_address[1]))
srv.serve_forever()
