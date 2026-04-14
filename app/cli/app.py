"""CLI 入口：组装模型、沙箱与 Agent。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from app.agents.research import create_research_agent
from app.cli.parser import parse_args
from app.models.resolver import ensure_provider_key, require_env, resolve_chat_model
from app.sandbox.factory import (
    build_backend,
    resolve_sandbox_mode,
    verify_sandbox_backend,
)
from app.streaming.console import run_turn_streaming


def main() -> None:
    load_dotenv()
    args = parse_args()
    sandbox_mode = resolve_sandbox_mode(args.sandbox)
    workspace = Path(args.sandbox_workspace).resolve() if args.sandbox_workspace else None
    backend = build_backend(
        sandbox_mode,
        workspace,
        docker_image=args.docker_image,
        docker_network=args.docker_network,
        docker_code_root=args.docker_code_root,
        docker_user_id=args.docker_user_id,
    )

    if args.verify_sandbox:
        sys.exit(verify_sandbox_backend(backend))

    require_env("TAVILY_API_KEY", "本示例使用 Tavily 作为联网搜索。")
    model = resolve_chat_model(args.model)
    ensure_provider_key(model)

    agent = create_research_agent(backend=backend, model=model)
    thread_cfg = {"configurable": {"thread_id": args.thread_id}}

    if args.chat:
        print(
            "多轮对话（空行或输入 /quit /exit 退出）。流式输出 + 工具过程见下方标记。"
            f" 沙箱: {sandbox_mode}"
        )
        first = args.query
        while True:
            try:
                line = first if first is not None else input("\n你: ").strip()
            except EOFError:
                print()
                break
            first = None
            if not line or line.lower() in {"/quit", "/exit", "exit", "quit"}:
                break
            run_turn_streaming(agent, line, thread_cfg)
        return

    query = (args.query or "").strip() or "搜索武汉最近的新闻"
    run_turn_streaming(agent, query, thread_cfg)
