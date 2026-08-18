#!/usr/bin/env python3
"""
test_mcp_model.py - MCP agent test (SSE / network transport).

The photo_mcp_server now runs as a standalone network service
(``FastMCP("photo-mcp-server", host="0.0.0.0", port=8000)`` +
``mcp.run(transport="sse")``) so a remote vLLM instance can reach it. This test:
  1. Launches photo_mcp_server.py as a background process (SSE on :8000).
  2. Connects to it with a small raw JSON-RPC-over-SSE client (see
     RawSseClient below).
  3. Drives a local OpenAI-compatible model (default :8001) to USE the tools.
  4. Runs the agent loop: the model decides which tools to call, the client
     calls them on the MCP server, results are fed back, and the model returns
     a final answer.

This is a genuine end-to-end MCP + LLM test over a network transport: the model
decides which tools to call, and the results come from the real
photo_mcp_server (not the model guessing).

Why a raw client instead of ``mcp.ClientSession``:
  With this FastMCP 1.29.0 server the high-level ``mcp.ClientSession`` fails the
  SSE handshake (``McpError: Invalid request parameters``), while a raw
  JSON-RPC client over the same SSE endpoint works. So the test talks to the
  server directly: it opens the ``/sse`` stream to learn the ``/messages/``
  endpoint, performs the initialize handshake, then sends ``tools/list`` and
  ``tools/call`` requests and reads the answers back on the SSE stream.

Usage:
  python3 test_mcp_model.py                 # first 3 images
  python3 test_mcp_model.py --limit 5
  python3 test_mcp_model.py --all
  python3 test_mcp_model.py --mcp-url http://127.0.0.1:8000/sse \
                             --model-url http://127.0.0.1:8001/v1/chat/completions \
                             --model-name Inferact/Qwen3.8-27B-NVFP4
"""

import argparse
import asyncio
import glob
import json
import os
import sys
import time
from urllib.parse import urlsplit

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
MCP_SCRIPT = os.path.join(HERE, "photo_mcp_server.py")
TEST_IMGS = os.path.join(HERE, "test-imgs")

DEFAULT_MCP_URL = "http://127.0.0.1:8000/sse"
DEFAULT_MODEL_URL = "http://127.0.0.1:8001/v1/chat/completions"
DEFAULT_MODEL_NAME = "Inferact/Qwen3.8-27B-NVFP4"

SYSTEM_PROMPT = (
    "You are a photo-fixation analysis agent for 1D/2D codes "
    "(DataMatrix, QR, barcodes). You MUST use the provided tools to inspect "
    "the image; NEVER invent or guess a code's content. Only report values "
    "that the tools actually return. If no code is detected, say so clearly."
)


# --------------------------------------------------------------------------- #
# Raw JSON-RPC-over-SSE MCP client
# --------------------------------------------------------------------------- #
class RawSseClient:
    """A minimal MCP client that speaks JSON-RPC over the SSE transport.

    The server exposes ``/sse`` (GET) which, on connect, emits an ``endpoint``
    event carrying the ``/messages/`` POST URL; JSON-RPC requests go there and
    their responses/notifications come back on the same SSE stream. This class
    is intentionally small and avoids ``mcp.ClientSession`` (which fails the
    handshake against this server).
    """

    def __init__(self, sse_url: str):
        self.sse_url = sse_url
        # The /messages/ endpoint is a relative path in the `endpoint` event;
        # resolve it against the scheme+host of the SSE URL.
        parts = urlsplit(sse_url)
        self._base = f"{parts.scheme}://{parts.netloc}"
        self._client: httpx.AsyncClient | None = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._messages_url: str | None = None
        self._reader: asyncio.Task | None = None
        self._next_id = 1

    async def connect(self, timeout: float = 30.0) -> dict:
        """Open the SSE stream, learn the messages URL, run the handshake."""
        self._client = httpx.AsyncClient(timeout=None)
        self._reader = asyncio.create_task(self._read())
        for _ in range(120):  # up to ~60s for the endpoint event
            if self._messages_url:
                break
            await asyncio.sleep(0.5)
        if not self._messages_url:
            raise RuntimeError(
                f"no /messages/ endpoint from SSE at {self.sse_url}")

        init = self._request(
            "initialize",
            {"protocolVersion": "2024-11-05",
             "capabilities": {},
             "clientInfo": {"name": "raw-sse-client", "version": "0.1"}})
        resp = await self._wait(init, timeout=timeout)
        # Complete the handshake with the initialized notification.
        await self._client.post(self._messages_url, json={
            "jsonrpc": "2.0", "method": "notifications/initialized"})
        return resp

    async def _read(self) -> None:
        """Consume the SSE stream; route JSON-RPC frames into the queue."""
        try:
            async with self._client.stream("GET", self.sse_url) as resp:
                event = None
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("event:"):
                        event = line[len("event:"):].strip()
                    elif line.startswith("data:"):
                        data = line[len("data:"):].strip()
                        if event == "endpoint":
                            self._messages_url = self._base + data
                        else:
                            try:
                                await self._queue.put(json.loads(data))
                            except Exception:
                                pass
                        event = None
        except Exception:
            pass

    def _request(self, method: str, params: dict | None = None) -> int:
        """Queue a JSON-RPC request; return its id (response arrives on SSE)."""
        i = self._next_id
        self._next_id += 1
        req = {"jsonrpc": "2.0", "id": i, "method": method}
        if params is not None:
            req["params"] = params
        asyncio.create_task(
            self._client.post(self._messages_url, json=req)
            if self._messages_url else asyncio.sleep(0))
        return i

    async def _wait(self, id: int, timeout: float = 60.0) -> dict:
        """Return the JSON-RPC response with the matching id."""
        while True:
            msg = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            if msg.get("id") == id:
                return msg

    async def list_tools(self) -> list:
        i = self._request("tools/list", {})
        resp = await self._wait(i, timeout=30)
        return resp.get("result", {}).get("tools", [])

    async def call_tool(self, name: str, args: dict) -> str:
        """Call a tool and return its text result (or a tool_error JSON)."""
        i = self._request("tools/call", {"name": name, "arguments": args})
        try:
            resp = await self._wait(i, timeout=120)
        except asyncio.TimeoutError:
            return json.dumps({"status": "tool_error",
                              "message": f"timeout calling {name}"})
        if "error" in resp:
            return json.dumps({"status": "tool_error",
                              "message": str(resp["error"])})
        result = resp.get("result", {})
        parts = []
        for block in result.get("content", []):
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            else:
                parts.append(str(block))
        text = "\n".join(parts) if parts else "(empty result)"
        if result.get("isError"):
            return json.dumps({"status": "tool_error", "message": text})
        return text

    async def close(self) -> None:
        if self._reader:
            self._reader.cancel()
        if self._client:
            await self._client.aclose()


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #
async def _launch_server():
    """Start photo_mcp_server.py as a background process (SSE on :8000)."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-u", MCP_SCRIPT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    # Drain the server's stderr so a full pipe buffer can't block it.
    asyncio.create_task(_drain(proc.stderr))
    return proc


async def _drain(stream):
    try:
        while True:
            line = await stream.readline()
            if not line:
                break
    except Exception:
        pass


def _terminate(proc):
    try:
        if proc is not None and proc.returncode is None:
            proc.terminate()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Model + agent
# --------------------------------------------------------------------------- #
def _to_openai_tool(tool):
    """Convert an MCP tool definition into an OpenAI function schema."""
    schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", "") or "",
            "parameters": schema,
        },
    }


async def _call_model(client, model_url, model_name, messages, tools,
                      temperature=0.0, max_tokens=2048):
    """One chat-completions call to the OpenAI-compatible model endpoint."""
    payload = {
        "model": model_name,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = await client.post(model_url, json=payload, timeout=600.0)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]


async def _call_mcp_tool(client, name, args):
    """Execute a tool on the MCP server via the raw client and return text."""
    try:
        return await client.call_tool(name, args)
    except Exception as e:  # never crash the agent loop on a tool error
        return json.dumps({"status": "tool_error", "message": str(e)})


async def _run_agent(mcp, model_client, model_url, model_name, image_path,
                     tools, max_rounds=6):
    """Run the tool-calling agent loop for a single image."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Analyze this image: {image_path}\n"
            "Use the available tools to detect any 1D/2D codes and decode "
            "them. Start with detect_code, then decode_code (or "
            "align_perspective -> decode_code) as appropriate. Report the "
            "decoded values, or state clearly that no code was found."
        )},
    ]
    log = []
    for rnd in range(1, max_rounds + 1):
        t0 = time.time()
        msg = await _call_model(model_client, model_url, model_name,
                                messages, tools)
        dt = round(time.time() - t0, 2)
        messages.append(msg)

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            content = msg.get("content") or ""
            log.append({"round": rnd, "type": "final", "content": content,
                        "seconds": dt})
            return {"final": content, "log": log}

        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            raw = fn.get("arguments", "{}")
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except Exception:
                args = {}
            result = await _call_mcp_tool(mcp, name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": result,
            })
            log.append({"round": rnd, "type": "tool", "name": name,
                        "args": args, "result": result, "seconds": dt})

    return {"final": None, "log": log}


# --------------------------------------------------------------------------- #
# Connect + run
# --------------------------------------------------------------------------- #
async def _connect_and_run(args, selected):
    """Connect to the MCP SSE server (retry until ready) and run the agent."""
    mcp = RawSseClient(args.mcp_url)
    deadline = time.time() + 90
    while True:
        try:
            await mcp.connect(timeout=30)
            break
        except Exception:
            if time.time() > deadline:
                raise RuntimeError(
                    f"MCP server not reachable at {args.mcp_url}")
            await asyncio.sleep(1.0)

    tools = [_to_openai_tool(t) for t in await mcp.list_tools()]
    print(f"MCP tools available: "
          f"{[t['function']['name'] for t in tools]}")
    async with httpx.AsyncClient() as model_client:
        for i, image_path in enumerate(selected, 1):
            name = os.path.basename(image_path)
            print(f"\n[{i}/{len(selected)}] {name}")
            t0 = time.time()
            result = await _run_agent(mcp, model_client, args.model_url,
                                      args.model_name, image_path, tools,
                                      max_rounds=args.max_rounds)
            for entry in result["log"]:
                if entry["type"] == "tool":
                    print(f"  tool #{entry['round']} {entry['name']}"
                          f"({entry['args']}) [{entry['seconds']}s]\n"
                          f"     -> {entry['result'][:300]}")
                elif entry["type"] == "final":
                    print(f"  FINAL ({entry['seconds']}s): "
                          f"{entry['content'][:400]}")
            print(f"  -> total {round(time.time() - t0, 1)}s")
    await mcp.close()


async def _amain():
    ap = argparse.ArgumentParser(
        description="MCP agent test: local model + photo_mcp_server (SSE)")
    ap.add_argument("--mcp-url", default=DEFAULT_MCP_URL,
                    help="MCP SSE endpoint (default http://127.0.0.1:8000/sse)")
    ap.add_argument("--model-url", default=DEFAULT_MODEL_URL)
    ap.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--limit", type=int, default=3,
                    help="number of images to process (default 3)")
    ap.add_argument("--all", action="store_true",
                    help="process every image in test-imgs")
    ap.add_argument("--max-rounds", type=int, default=6,
                    help="max tool-calling rounds per image (default 6)")
    args = ap.parse_args()

    imgs = sorted(glob.glob(os.path.join(TEST_IMGS, "*.jpg"))
                  + glob.glob(os.path.join(TEST_IMGS, "*.png")))
    # Exclude align byproducts (created by align_perspective) from the inputs.
    imgs = [p for p in imgs if not p.endswith("_aligned.png")]
    selected = imgs if args.all else imgs[:args.limit]

    if not selected:
        print(f"No images found in {TEST_IMGS}", file=sys.stderr)
        sys.exit(1)

    print(f"MCP server: {args.mcp_url}")
    print(f"Model     : {args.model_name} @ {args.model_url}")
    print(f"Images    : {len(selected)}/{len(imgs)} from {TEST_IMGS}")

    # Launch the MCP server (SSE) as a background process, then connect + run.
    proc = await _launch_server()
    try:
        await _connect_and_run(args, selected)
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        _terminate(proc)


if __name__ == "__main__":
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)