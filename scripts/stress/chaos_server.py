"""The mock RAG server behind fault injection, logging every /query attempt.

    STRESS_LOG=out/stress/server.log uvicorn scripts.stress.chaos_server:app --port 8791

- latency: 20-200ms jitter, 1% of requests stall 2s
- periodic 429s: for 1s out of every 8s, everything gets 429 Retry-After: 1
- random 429s: 3% of the rest get 429 Retry-After: 0.25
"""

import asyncio
import json
import os
import random
import time

from src.connectors.mock_rag_server import create_app

inner = create_app(os.environ.get("UMBRA_KB_PATH", "data/synthetic_kb"))
log = open(os.environ["STRESS_LOG"], "a", buffering=1)
rng = random.Random(1)
t0 = time.time()


async def app(scope, receive, send):
    if scope["type"] != "http" or scope["path"] != "/query":
        return await inner(scope, receive, send)
    body = b""
    while True:
        msg = await receive()
        body += msg.get("body", b"")
        if msg["type"] == "http.disconnect":
            return  # the proxy cut the connection between headers and body
        if not msg.get("more_body"):
            break
    q = json.loads(body)["query"]
    now = time.time()
    if (now - t0) % 8 < 1:
        outcome, retry_after = "429_burst", "1"
    elif rng.random() < 0.03:
        outcome, retry_after = "429_random", "0.25"
    else:
        outcome, retry_after = "served", None
    log.write(json.dumps({"t": now, "q": q, "outcome": outcome}) + "\n")
    if retry_after:
        await send({"type": "http.response.start", "status": 429,
                    "headers": [(b"retry-after", retry_after.encode()), (b"content-type", b"application/json")]})
        return await send({"type": "http.response.body", "body": b'{"detail":"rate limited"}'})
    await asyncio.sleep(2.0 if rng.random() < 0.01 else rng.uniform(0.02, 0.2))

    async def replay():
        return {"type": "http.request", "body": body, "more_body": False}

    await inner(scope, replay, send)
