"""Deep Agents 官方快速入门示例（研究 + Tavily 联网搜索）。

- 默认单轮：流式打印助手回复，并在控制台标注工具调用与工具输出。
- 多轮：`python main.py --chat`（可用 `--query` 作为首轮输入，`--thread-id` 区分会话）。
- 沙箱后端：`--sandbox local` 使用 `LocalShellBackend`，启用内置 `execute` 工具（在宿主机执行命令，仅限可信环境）；`ls`/`execute` 的「当前目录」默认是**本仓库根目录**（`main.py` 所在目录），不是终端里的 `cwd`。需要隔离空目录时可传 `--sandbox-workspace`。
- 自检：`python main.py --verify-sandbox`（可不配 Tavily；`local` 模式下直接调用 `execute` 验证）。

文档: https://docs.langchain.com/oss/python/deepagents/quickstart

Minimax（官方 OpenAI 兼容，推荐）:
  export OPENAI_BASE_URL=https://api.minimaxi.com/v1
  export OPENAI_API_KEY=<你的 API Key>
  模型名可用 MINIMAX_MODEL 或 OPENAI_MODEL（默认 MiniMax-M2.7）

其它方式见 .env.example。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_community.chat_models import MiniMaxChat
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Overwrite

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend, StateBackend
from deepagents.backends.protocol import BackendProtocol, SandboxBackendProtocol
from tavily import TavilyClient


DOCS_URL = "https://docs.langchain.com/oss/python/deepagents/quickstart"

RESEARCH_INSTRUCTIONS = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""

# local 模式虚拟根目录：默认同项目根，避免 .deepagent_sandbox_workspace 空目录导致 ls 一直为空
SANDBOX_LOCAL_DEFAULT_ROOT = Path(__file__).resolve().parent
SandboxMode = Literal["state", "local"]

_PROVIDER_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "baseten": "BASETEN_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}


def _require_env(name: str, hint: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"错误: 未设置环境变量 {name}。{hint}", file=sys.stderr)
        print(f"说明见: {DOCS_URL}", file=sys.stderr)
        sys.exit(1)
    return value


def _default_openai_compat_model() -> str:
    return (
        os.environ.get("OPENAI_MODEL", "").strip()
        or os.environ.get("MINIMAX_MODEL", "").strip()
        or "MiniMax-M2.7"
    )


def _build_openai_compatible_chat(model_name: str) -> ChatOpenAI:
    base = _require_env(
        "OPENAI_BASE_URL",
        "使用 Minimax OpenAI 兼容接口时需设置（官方示例: https://api.minimaxi.com/v1）。",
    )
    key = _require_env("OPENAI_API_KEY", "使用 OpenAI 兼容接口时需设置。")
    return ChatOpenAI(
        model=model_name,
        base_url=base.rstrip("/"),
        api_key=key,
    )


def _minimax_model_name_from_spec(spec: str) -> str:
    name = spec.split(":", 1)[1].strip()
    return name or _default_openai_compat_model()


def resolve_chat_model(cli_model: str | None) -> str | BaseChatModel | None:
    """返回 create_deep_agent 的 model：字符串、ChatOpenAI、MiniMaxChat 或 None（库默认 Anthropic）。"""
    spec = (cli_model or os.environ.get("DEEPAGENTS_MODEL", "").strip() or None)

    if spec and spec.lower().startswith("minimax:"):
        name = _minimax_model_name_from_spec(spec)
        if os.environ.get("OPENAI_BASE_URL", "").strip() and os.environ.get(
            "OPENAI_API_KEY", ""
        ).strip():
            return _build_openai_compatible_chat(name)
        _require_env("MINIMAX_API_KEY", "未配置 OpenAI 兼容接口时，使用 Minimax 原生接口需要 MINIMAX_API_KEY。")
        return MiniMaxChat(model=name)

    if spec:
        return spec

    if os.environ.get("OPENAI_BASE_URL", "").strip() and os.environ.get(
        "OPENAI_API_KEY", ""
    ).strip():
        return _build_openai_compatible_chat(_default_openai_compat_model())

    if os.environ.get("MINIMAX_API_KEY", "").strip():
        return MiniMaxChat(
            model=os.environ.get("MINIMAX_MODEL", "abab6.5s-chat").strip()
        )

    return None


def _ensure_provider_key(model: str | BaseChatModel | None) -> None:
    if isinstance(model, ChatOpenAI):
        _require_env("OPENAI_API_KEY", "当前使用 OpenAI 兼容客户端（如 Minimax）。")
        _require_env("OPENAI_BASE_URL", "当前使用 OpenAI 兼容客户端（如 Minimax）。")
        return
    if isinstance(model, MiniMaxChat):
        _require_env("MINIMAX_API_KEY", "当前使用 Minimax 原生 HTTP 接口。")
        return
    if model is None:
        _require_env(
            "ANTHROPIC_API_KEY",
            "未配置 OPENAI_BASE_URL+OPENAI_API_KEY 或 MINIMAX_API_KEY，默认使用 Anthropic。",
        )
        return
    prefix = model.split(":", 1)[0].strip().lower()
    if prefix == "ollama":
        return
    env_name = _PROVIDER_ENV.get(prefix)
    if env_name:
        _require_env(env_name, f"当前模型为 {model}。")


def build_internet_search_tool(api_key: str):
    tavily_client = TavilyClient(api_key=api_key)

    def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
    ):
        """Run a web search."""
        return tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )

    return internet_search


def resolve_sandbox_mode(cli: str | None) -> SandboxMode:
    raw = (cli or os.environ.get("DEEPAGENTS_SANDBOX") or "state").strip().lower()
    if raw in ("state", "memory", "default"):
        return "state"
    if raw in ("local", "shell", "localshell"):
        return "local"
    print(
        f"错误: 无效的沙箱模式 {raw!r}，请使用 state 或 local（或环境变量 DEEPAGENTS_SANDBOX）。",
        file=sys.stderr,
    )
    sys.exit(1)


def build_backend(mode: SandboxMode, workspace: Path | None) -> BackendProtocol:
    if mode == "state":
        return StateBackend()
    root = workspace if workspace is not None else SANDBOX_LOCAL_DEFAULT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return LocalShellBackend(
        root_dir=str(root),
        virtual_mode=True,
        inherit_env=True,
    )


def verify_sandbox_backend(backend: BackendProtocol) -> int:
    """直接验证后端是否支持并正确执行 shell（不调用 LLM）。"""
    if not isinstance(backend, SandboxBackendProtocol):
        print(
            "沙箱验证: 当前为 StateBackend（默认）。"
            "FilesystemMiddleware 不会注册 execute 工具；此为预期行为。"
        )
        return 0
    r = backend.execute("echo DEEPAGENT_SANDBOX_OK")
    if r.exit_code != 0 or "DEEPAGENT_SANDBOX_OK" not in (r.output or ""):
        print(
            f"沙箱验证失败: exit_code={r.exit_code!r} output={r.output!r}",
            file=sys.stderr,
        )
        return 1
    r2 = backend.execute('python3 -c "print(40+2)"')
    if r2.exit_code != 0 or "42" not in (r2.output or ""):
        print(
            f"沙箱验证失败（python3）: exit_code={r2.exit_code!r} output={r2.output!r}",
            file=sys.stderr,
        )
        return 1
    print(
        "沙箱验证通过: execute(echo) 与 execute(python3) 均成功。"
        f" 工作目录后端 id={backend.id!r}。"
    )
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep Agents 快速入门：研究型 Agent + Tavily")
    p.add_argument(
        "--query",
        default=None,
        help="单轮模式下的用户问题；与 --chat 联用时为首次输入（可选）",
    )
    p.add_argument(
        "--chat",
        action="store_true",
        help="控制台多轮对话（需同一 thread 记忆，依赖 checkpointer）",
    )
    p.add_argument(
        "--thread-id",
        default="cli",
        help="多轮对话线程 ID（同一 ID 共享上下文）",
    )
    p.add_argument(
        "--model",
        default=None,
        help="覆盖 DEEPAGENTS_MODEL；Minimax 原生写法 minimax:模型名；否则如 openai:gpt-4o",
    )
    p.add_argument(
        "--sandbox",
        default=None,
        choices=["state", "local"],
        help="后端模式：state=状态内文件（默认）；local=本机 LocalShellBackend，启用 execute（有风险，仅可信环境）",
    )
    p.add_argument(
        "--sandbox-workspace",
        default=None,
        metavar="DIR",
        help="--sandbox local 时的虚拟根目录（默认同 main.py 所在项目根；可改为子目录以限制可见文件）",
    )
    p.add_argument(
        "--verify-sandbox",
        action="store_true",
        help="仅验证沙箱 execute 是否可用，不调用模型（无需 TAVILY_API_KEY）",
    )
    return p.parse_args()


def _truncate(s: str, max_len: int) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def _serialize_tool_args(args: Any) -> str:
    if isinstance(args, str):
        return _truncate(args, 800)
    try:
        return _truncate(json.dumps(args, ensure_ascii=False), 800)
    except (TypeError, ValueError):
        return _truncate(str(args), 800)


def _extract_text_delta(msg: BaseMessage) -> str:
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


def _messages_from_update_payload(raw: Any) -> list[Any]:
    """updates 里 messages 可能是 list，也可能是 Overwrite(value=[...])。"""
    if raw is None:
        return []
    if isinstance(raw, Overwrite):
        raw = raw.value
    if isinstance(raw, list):
        return raw
    if isinstance(raw, BaseMessage):
        return [raw]
    return []


def _print_tool_events_from_updates(update: dict[str, Any]) -> bool:
    """从 LangGraph updates 中打印本轮新增的模型工具调用与工具结果。返回是否打印过。"""
    printed = False
    for node, payload in update.items():
        if node.startswith("__"):
            continue
        if not isinstance(payload, dict):
            continue
        messages = _messages_from_update_payload(payload.get("messages"))
        if not messages:
            continue
        for m in messages:
            if isinstance(m, AIMessage) and m.tool_calls:
                for tc in m.tool_calls:
                    name = tc.get("name", "?")
                    args = tc.get("args")
                    if args is None and "arguments" in tc:
                        args = tc["arguments"]
                    print(f"\n[工具调用] 节点={node} 工具={name} 参数={_serialize_tool_args(args)}")
                    printed = True
            elif isinstance(m, ToolMessage):
                body = str(m.content)
                print(f"\n[工具结果] 节点={node} 工具={m.name} 输出={_truncate(body, 1200)}")
                printed = True
    return printed


def run_turn_streaming(agent, user_text: str, config: dict) -> None:
    """流式输出助手文本，并在 updates 中打印工具调用与工具返回。"""
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
            text = _extract_text_delta(msg)
            if text:
                if not in_assistant_stream or saw_tool_since_text:
                    sys.stdout.write("\n助手: ")
                    sys.stdout.flush()
                    saw_tool_since_text = False
                    in_assistant_stream = True
                sys.stdout.write(text)
                sys.stdout.flush()
        elif part["type"] == "updates":
            if _print_tool_events_from_updates(part["data"]):
                saw_tool_since_text = True
    if in_assistant_stream:
        print()
    print()


def main() -> None:
    load_dotenv()
    args = parse_args()
    sandbox_mode = resolve_sandbox_mode(args.sandbox)
    workspace = Path(args.sandbox_workspace).resolve() if args.sandbox_workspace else None
    backend = build_backend(sandbox_mode, workspace)

    if args.verify_sandbox:
        sys.exit(verify_sandbox_backend(backend))

    _require_env("TAVILY_API_KEY", "本示例使用 Tavily 作为联网搜索。")
    model = resolve_chat_model(args.model)
    _ensure_provider_key(model)

    internet_search = build_internet_search_tool(os.environ["TAVILY_API_KEY"])
    checkpointer = InMemorySaver()
    agent = create_deep_agent(
        model=model,
        tools=[internet_search],
        system_prompt=RESEARCH_INSTRUCTIONS,
        checkpointer=checkpointer,
        backend=backend,
    )

    thread_cfg = {"configurable": {"thread_id": args.thread_id}}

    if args.chat:
        print("多轮对话（空行或输入 /quit /exit 退出）。流式输出 + 工具过程见下方标记。")
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


if __name__ == "__main__":
    main()
