"""控制台流式输出与工具调用日志（带去重）。"""

from __future__ import annotations

import json
import sys
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langgraph.types import Overwrite


def truncate(s: str, max_len: int) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def serialize_tool_args(args: Any) -> str:
    if isinstance(args, str):
        return truncate(args, 800)
    try:
        return truncate(json.dumps(args, ensure_ascii=False), 800)
    except (TypeError, ValueError):
        return truncate(str(args), 800)


def extract_text_delta(msg: BaseMessage) -> str:
    if not isinstance(msg, AIMessageChunk):
        return ""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return ""


def messages_from_update_payload(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, Overwrite):
        raw = raw.value
    if isinstance(raw, list):
        return raw
    if isinstance(raw, BaseMessage):
        return [raw]
    return []


def message_tool_log_key(m: BaseMessage) -> str | None:
    msg_id = getattr(m, "id", None)
    if msg_id:
        return f"id:{msg_id}"
    if isinstance(m, ToolMessage):
        return f"tool:{m.tool_call_id}:{m.name}"
    if isinstance(m, AIMessage) and m.tool_calls:
        parts: list[tuple[str, str, str]] = []
        for tc in m.tool_calls:
            if not isinstance(tc, dict):
                continue
            tid = str(tc.get("id") or "")
            name = str(tc.get("name") or "")
            args = tc.get("args")
            if args is None:
                args = tc.get("arguments")
            try:
                arg_s = json.dumps(args, sort_keys=True, default=str)
            except (TypeError, ValueError):
                arg_s = str(args)
            parts.append((tid, name, arg_s))
        return f"ai:{tuple(parts)}"
    return None


def seed_tool_log_keys_from_prior_messages(msgs: list[Any]) -> set[str]:
    keys: set[str] = set()
    for m in msgs:
        if isinstance(m, BaseMessage):
            k = message_tool_log_key(m)
            if k:
                keys.add(k)
    return keys


def print_tool_events_from_updates(
    update: dict[str, Any],
    seen_keys: set[str],
) -> bool:
    printed = False
    for node, payload in update.items():
        if node.startswith("__"):
            continue
        if not isinstance(payload, dict):
            continue
        messages = messages_from_update_payload(payload.get("messages"))
        if not messages:
            continue
        for m in messages:
            if not isinstance(m, BaseMessage):
                continue
            key = message_tool_log_key(m)
            if key is None or key in seen_keys:
                continue
            if isinstance(m, AIMessage) and m.tool_calls:
                for tc in m.tool_calls:
                    name = tc.get("name", "?")
                    args = tc.get("args")
                    if args is None and "arguments" in tc:
                        args = tc["arguments"]
                    print(f"\n[工具调用] 节点={node} 工具={name} 参数={serialize_tool_args(args)}")
                    printed = True
                seen_keys.add(key)
            elif isinstance(m, ToolMessage):
                body = str(m.content)
                print(f"\n[工具结果] 节点={node} 工具={m.name} 输出={truncate(body, 1200)}")
                printed = True
                seen_keys.add(key)
    return printed


def run_turn_streaming(agent: Any, user_text: str, config: dict) -> None:
    """流式输出助手文本，并在 updates 中打印本回合新增的工具调用与工具返回（不重复打印历史）。"""
    try:
        snap = agent.get_state(config)
        vals = snap.values
        if isinstance(vals, dict):
            raw_prev = vals.get("messages")
        else:
            raw_prev = getattr(vals, "messages", None) if vals is not None else None
        prev_msgs: list[Any] = list(raw_prev) if isinstance(raw_prev, list) else []
    except Exception:
        prev_msgs = []
    seen_tool_log_keys = seed_tool_log_keys_from_prior_messages(prev_msgs)

    payload = {"messages": [{"role": "user", "content": user_text}]}
    in_assistant_stream = False
    saw_tool_since_text = False
    for part in agent.stream(
        payload,
        config=config,
        stream_mode=["messages", "updates"],
        subgraphs=True,
        version="v2",
    ):
        if part["type"] == "messages":
            msg, _meta = part["data"]
            text = extract_text_delta(msg)
            if text:
                if not in_assistant_stream or saw_tool_since_text:
                    sys.stdout.write("\n助手: ")
                    sys.stdout.flush()
                    saw_tool_since_text = False
                    in_assistant_stream = True
                sys.stdout.write(text)
                sys.stdout.flush()
        elif part["type"] == "updates":
            if print_tool_events_from_updates(part["data"], seen_tool_log_keys):
                saw_tool_since_text = True
    if in_assistant_stream:
        print()
    print()
