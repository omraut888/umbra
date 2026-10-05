"""TCP proxy that resets the connection on ~4% of client writes, before forwarding them.

    python scripts/stress/drop_proxy.py 8790 8791 out/stress/proxy.log

A request's headers and body often arrive in separate reads, so most drops
can't be tied to a query; those are logged with "q": null.
"""

import asyncio
import json
import random
import sys
import time

listen, target, log_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
log = open(log_path, "a", buffering=1)
rng = random.Random(2)


def query_of(chunk: bytes):
    try:
        return json.loads(chunk.split(b"\r\n\r\n", 1)[1])["query"]
    except (IndexError, ValueError, KeyError):
        return None


async def pipe(reader, writer, other, drop: bool):
    try:
        while data := await reader.read(65536):
            if drop and rng.random() < 0.04:
                log.write(json.dumps({"t": time.time(), "q": query_of(data), "outcome": "conn_drop"}) + "\n")
                writer.transport.abort()
                other.transport.abort()
                return
            other.write(data)
            await other.drain()
    except OSError:
        pass
    finally:
        writer.close()
        other.close()


async def handle(client_r, client_w):
    server_r, server_w = await asyncio.open_connection("127.0.0.1", target)
    await asyncio.gather(pipe(client_r, client_w, server_w, True), pipe(server_r, server_w, client_w, False))


async def main():
    server = await asyncio.start_server(handle, "127.0.0.1", listen, backlog=1024)
    async with server:
        await server.serve_forever()


asyncio.run(main())
