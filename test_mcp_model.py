#!/usr/bin/env python3
"""
test_mcp_model.py - MCP agent test.

Drives a local OpenAI-compatible model (default http://127.0.0.1:8001, model
Inferact/Qwen3.8-27B-NVFP4) to USE the photo_mcp_server tools
(detect_code / align_perspective / decode_code) over images in ./test-imgs.

Pipeline per image:
  1. A raw JSON-RPC client launches photo_mcp_server.py (stdio transport) and
     performs the MCP initialize handshake, then lists the tools.
  2. The tool schemas are converted to OpenAI function-call format.
  3. The model is asked to analyze the image using the tools (agent loop).
  4. Every tool_call is executed against the MCP server (tools/call); results
     are fed back to the model.
  5. The model returns a final answer (decoded values / "no code found").

A raw JSON-RPC client is used (instead of the higher-level MCP ClientSession)
because it reliably completes the handshake with this FastMCP server.

This is a genuine end-to-end MCP + LLM test: the model decides which tools to
call, and the results come from the real photo_mcp_server (not the model
guessing).

Usage:
  python3 test_mcp_model.py                 # first 3 images
  python3 test_mcp_model.py --limit 5
  python3 test_mcp_model.py --all
  python3 test_mcp_model.py --model-url http://127.0.0.1:8001/v1/chat/completions \
                             --model-name Inferact/Qwen3.8-27B-NVFP4
"""

import argparse
import asyncio
import glob
import json
import os
import sys
import time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
MCP_SCRIPT = os.path.join(HERE, "photo_mcp_server.py")
TEST_IMGS = os.path.join(HERE, "test-imgs")

DEFAULT_MODEL_URL = "http://127.0.0.1:8001/v1/chat/completions"
DEFAULT_MODEL_NAME = "Inferact/Qwen3.8-27B-NVFP4"
PROTOCOL_VERSION = "2024-11-05"

SYSTEM_PROMPT = (
    "You are a photo-fixation analysis agent for 1D/2D codes "
    "(DataMatrix, QR, barcodes). You MUST use the provided tools to inspect "
    "the image; NEVER invent or guess a code's content. Only report values "
    "that the tools actually return. If no code is detected, say so clearly."
)


# --------------------------------------------------------------------------- #
# Raw JSON-RPC client over stdio (reliable handshake with FastMCP server)
# --------------------------------------------------------------------------- #
class MCPRawClient:
    """Minimal async MCP client speaking JSON-RPC over the server's stdio."""

    def __init__(self, script, cwd):
        self.script = script
        self.cwd = cwd
        self.proc = None
        self._id = 0
        self._pending = {}
        self.tools = []

    async def connect(self, timeout=60):
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable, self.script, cwd=self.cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        asyncio.create_task(self._drain_stderr())
        asyncio.create_task(self._read_loop())
        await self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "test_mcp_model", "version": "1.0"},
        }, timeout=timeout)
        await self.notify("notifications/initialized")
        self.tools = await self.list_tools()
        return self.tools

    async def _drain_stderr(self):
        try:
            while True:
                line = await self.proc.stderr.readline()
                if not line:
                    break
        except Exception:
            pass

    async def _read_loop(self):
        try:
            while True:
                line = await self.proc.stdout.readline()
                if not line:
                    break
                txt = line.decode().strip()
                if not txt:
                    continue
                try:
                    msg = json.loads(txt)
                except Exception:
                    continue
                rid = msg.get("id")
                if rid is not None:
                    fut = self._pending.get(rid)
                    if fut and not fut.done():
                        fut.set_result(msg)
        except Exception:
            pass

    async def request(self, method, params=None, timeout=120):
        self._id += 1
        rid = self._id
        req = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            req["params"] = params
        self.proc.stdin.write(json.dumps(req).encode() + b"\n")
        await self.proc.stdin.drain()
        fut = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        resp = await asyncio.wait_for(fut, timeout=timeout)
        if "error" in resp:
            raise RuntimeError(f"{method} error: {resp['error']}")
        return resp.get("result", {})

    async def notify(self, method, params=None):
        req = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            req["params"] = params
        self.proc.stdin.write(json.dumps(req).encode() + b"\n")
        await self.proc.stdin.drain()

    async def list_tools(self):
        result = await self.request("tools/list", {})
        return result.get("tools", [])

    async def call_tool(self, name, args):
        result = await self.request("tools/call", {
            "name": name, "arguments": args,
        })
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

    async def close(self):
        try:
            self.proc.terminate()
        except Exception:
            pass


def _to_openai_tool(tool):
    """Convert an MCP tool definition into an OpenAI function schema."""
    schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
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


async def _run_agent(mcp, client, model_url, model_name, image_path,
                     max_rounds=6):
    """Run the tool-calling agent loop for a single image."""
    tools = [_to_openai_tool(t) for t in mcp.tools]

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
        msg = await _call_model(client, model_url, model_name, messages, tools)
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
            try:
                result = await mcp.call_tool(name, args)
            except Exception as e:
                result = json.dumps({"status": "tool_error", "message": str(e)})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": result,
            })
            log.append({"round": rnd, "type": "tool", "name": name,
                        "args": args, "result": result, "seconds": dt})

    return {"final": None, "log": log}


async def _amain():
    ap = argparse.ArgumentParser(
        description="MCP agent test: local model + photo_mcp_server tools")
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
    selected = imgs if args.all else imgs[:args.limit]

    if not selected:
        print(f"No images found in {TEST_IMGS}", file=sys.stderr)
        sys.exit(1)

    print(f"Model : {args.model_name} @ {args.model_url}")
    print(f"Images: {len(selected)}/{len(imgs)} from {TEST_IMGS}")

    # Launch the MCP server once (it downloads models on first tool call).
    mcp = MCPRawClient(MCP_SCRIPT, HERE)
    try:
        tools = await mcp.connect()
    except Exception as e:
        print(f"Failed to connect to MCP server: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"MCP tools available: {[t.get('name') for t in tools]}")

    async with httpx.AsyncClient() as client:
        for i, image_path in enumerate(selected, 1):
            name = os.path.basename(image_path)
            print(f"\n{'=' * 70}\n[{i}/{len(selected)}] {name}")
            t0 = time.time()
            try:
                outcome = await _run_agent(
                    mcp, client, args.model_url, args.model_name,
                    image_path, max_rounds=args.max_rounds)
            except Exception as e:
                print(f"  ERROR: {e}")
                outcome = {"final": None, "log": []}
            dt = round(time.time() - t0, 1)

            for step in outcome["log"]:
                if step["type"] == "tool":
                    res = step["result"]
                    if len(res) > 300:
                        res = res[:300] + " ...[truncated]"
                    print(f"  tool #{step['round']} {step['name']}("
                          f"{step['args']}) [{step['seconds']}s]")
                    print(f"     -> {res}")
                else:
                    print(f"  final  #{step['round']} [{step['seconds']}s]")

            final = outcome["final"]
            if final:
                print(f"  RESULT: {final}")
            else:
                print(f"  RESULT: (no final answer within "
                      f"{args.max_rounds} rounds)")
            print(f"  elapsed: {dt}s")

    print(f"\n{'=' * 70}\nDone.")


def main():
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()