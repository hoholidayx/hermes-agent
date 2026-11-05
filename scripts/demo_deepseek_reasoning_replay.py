#!/usr/bin/env python3
"""Live, bounded DeepSeek replay experiment; never reads or writes Hermes sessions.

Run through scripts/run-in-hermes-env with --env-file pointing to your credentials.
Every follow-up keeps tools and tool_choice=auto. Raw HTTP has no SDK retries,
padding, or Hermes sanitization that could hide a rejected request.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

TOOL_NAME = "lookup_demo_code"
TOOLS = [{"type": "function", "function": {
    "name": TOOL_NAME,
    "description": "Look up the demo code. The code is available only from this tool.",
    "parameters": {"type": "object", "properties": {"checksum": {"type": "integer"}},
                   "required": ["checksum"], "additionalProperties": False},
}}]
REASONING_KEYS = ("reasoning_content", "reasoning", "reasoning_details")
SCENARIOS = (
    "full_history",
    "strip_final_reasoning",
    "strip_all_reasoning",
    "drop_tool_exchange",
    "drop_tool_exchange_and_all_reasoning",
)
CHAT = "现在只进行普通聊天：请只回复‘你好’，不要调用任何工具。"


def summarize(messages: list[dict]) -> list[dict]:
    return [{
        "role": m["role"],
        "tool_calls": len(m.get("tool_calls") or []),
        "reasoning_chars": len(m.get("reasoning_content") or ""),
        "content_chars": len(m.get("content") or ""),
    } for m in messages]


def assistant_record(result: dict) -> dict:
    message = result["response"]["choices"][0]["message"]
    return {k: deepcopy(v) for k, v in message.items()
            if k in ("role", "content", "tool_calls", *REASONING_KEYS) and v is not None}


def followup_history(history: list[dict], scenario: str) -> list[dict]:
    messages = deepcopy(history)
    if scenario.startswith("drop_tool_exchange"):
        # Drop complete call/result pairs, never leave orphaned tool results.
        messages = [m for m in messages if m["role"] != "tool" and not m.get("tool_calls")]
    for message in messages:
        strip = "all_reasoning" in scenario or (
            scenario == "strip_final_reasoning" and not message.get("tool_calls")
        )
        if strip and message["role"] == "assistant":
            for key in REASONING_KEYS:
                message.pop(key, None)
    return [*messages, {"role": "user", "content": CHAT}]


def fresh_tool_ids(history: list[dict]) -> list[dict]:
    """Control for any server-side replay keyed by the original tool call IDs."""
    messages = deepcopy(history)
    ids = {call["id"]: f"call_00_demo_replay_{index:08d}"
           for index, call in enumerate(call for m in messages for call in m.get("tool_calls") or [])}
    for message in messages:
        for call in message.get("tool_calls") or []:
            call["id"] = ids[call["id"]]
        if message["role"] == "tool":
            message["tool_call_id"] = ids[message["tool_call_id"]]
    return messages


class Experiment:
    def __init__(self, args: argparse.Namespace, key: str, endpoint: str):
        self.args, self.key, self.endpoint = args, key, endpoint
        self.report = {
            "started_at": datetime.now(UTC).isoformat(),
            "endpoint": endpoint, "model": args.model, "calls": [],
        }

    def save(self) -> None:
        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        self.args.output.write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2).replace(self.key, "[REDACTED]"),
            encoding="utf-8",
        )

    def call(self, name: str, messages: list[dict]) -> dict:
        payload = {
            "model": self.args.model, "messages": messages, "tools": TOOLS,
            "tool_choice": "auto", "thinking": {"type": "enabled"},
            "reasoning_effort": self.args.effort, "max_tokens": self.args.max_tokens, "stream": False,
        }
        record = {"name": name, "request": deepcopy(payload), "message_shapes": summarize(messages)}
        self.report["calls"].append(record)
        self.save()
        print(f"START {name}", flush=True)
        started = time.monotonic()
        request = Request(self.endpoint, data=json.dumps(payload).encode(), headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
        })
        try:
            with urlopen(request, timeout=self.args.timeout) as response:
                record["status"] = response.status
                record["response"] = json.load(response)
                record["request_id"] = response.headers.get("x-request-id")
        except HTTPError as exc:
            record["status"] = exc.code
            record["error"] = exc.read().decode("utf-8", errors="replace").replace(self.key, "[REDACTED]")
        except (URLError, TimeoutError) as exc:
            record["status"] = None
            record["error"] = str(exc).replace(self.key, "[REDACTED]")
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if record["status"] == 200:
            message = assistant_record(record)
            record["returned_tool_calls"] = len(message.get("tool_calls") or [])
            record["finish_reason"] = record["response"]["choices"][0]["finish_reason"]
            record["chat_matches"] = (message.get("content") or "").strip("‘’'\"。！! \n") == "你好"
        self.save()
        print(json.dumps({k: record[k] for k in (
            "name", "status", "returned_tool_calls", "finish_reason", "chat_matches", "elapsed_seconds", "error",
        ) if k in record} | {"usage": record.get("response", {}).get("usage")}, ensure_ascii=False), flush=True)
        return record

    def seed(self) -> list[dict]:
        history = [{"role": "user", "content": (
            "先计算 (137×289+83×157) 除以 97 的余数，"
            f"将结果作为 checksum 调用 {TOOL_NAME}。"
            "工具会返回 code 和 a、b、c、d；收到结果后，再计算 (a×b+c×d) 除以 89 的余数。"
            "最终用一句话告诉我 code 和第二个余数。必须先自己算出 checksum，再调用工具。"
        )}]
        result = self.call("seed_tool_call", history)
        if result["status"] != 200:
            raise RuntimeError("Seed request failed; see report.")
        tool_turn = assistant_record(result)
        calls = tool_turn.get("tool_calls") or []
        if not calls or not tool_turn.get("reasoning_content"):
            raise RuntimeError("Seed did not produce a tool call with reasoning; experiment is inconclusive.")
        history.append(tool_turn)
        for call in calls:
            if call["function"]["name"] != TOOL_NAME:
                raise RuntimeError("Seed returned an unexpected tool.")
            history.append({"role": "tool", "tool_call_id": call["id"],
                            "content": json.dumps({"code": "DEMO-7319", "a": 283, "b": 719, "c": 41, "d": 53})})
        # This is a genuine same-user-turn continuation, with exact reasoning replay.
        result = self.call("seed_tool_result_replay", history)
        if result["status"] != 200 or result["returned_tool_calls"] or result["finish_reason"] != "stop":
            raise RuntimeError("Seed continuation did not complete; experiment is inconclusive.")
        history.append(assistant_record(result))
        if not history[-1].get("reasoning_content"):
            raise RuntimeError("Final seed reply has no reasoning; stripping plain-turn reasoning cannot be tested.")
        self.report["seed_history"] = deepcopy(history)
        self.save()
        return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--base-url")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--effort", choices=("low", "high", "max"), default="high")
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        parser.error("DEEPSEEK_API_KEY is missing; set it or pass --env-file.")
    base = (args.base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
    parts = urlsplit(base)
    if parts.scheme != "https" or parts.username or parts.password or parts.query or parts.fragment:
        parser.error("Use an HTTPS base URL without credentials, query, or fragment.")
    endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
    experiment = Experiment(args, key, endpoint)
    history = experiment.seed()
    same_turn = deepcopy(history[:-1])
    for message in same_turn:
        if message["role"] == "assistant":
            for key in REASONING_KEYS:
                message.pop(key, None)
    experiment.call("same_turn_strip_all_reasoning", same_turn)
    experiment.call("same_turn_full_reasoning_fresh_ids", fresh_tool_ids(history[:-1]))
    experiment.call("same_turn_strip_all_reasoning_fresh_ids", fresh_tool_ids(same_turn))
    experiment.call("next_turn_strip_all_reasoning_fresh_ids",
                    fresh_tool_ids(followup_history(history, "strip_all_reasoning")))
    for repeat in range(1, args.repeats + 1):
        for scenario in SCENARIOS:
            experiment.call(f"{scenario}#{repeat}", followup_history(history, scenario))
    print(f"REPORT {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
