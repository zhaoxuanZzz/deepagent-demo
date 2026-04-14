"""聊天模型解析与厂商 API Key 检查。"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from langchain_community.chat_models import MiniMaxChat
from langchain_openai import ChatOpenAI

from app.config import DOCS_URL

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

_PROVIDER_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "baseten": "BASETEN_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}


def require_env(name: str, hint: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"错误: 未设置环境变量 {name}。{hint}", file=sys.stderr)
        print(f"说明见: {DOCS_URL}", file=sys.stderr)
        sys.exit(1)
    return value


def default_openai_compat_model() -> str:
    return (
        os.environ.get("OPENAI_MODEL", "").strip()
        or os.environ.get("MINIMAX_MODEL", "").strip()
        or "MiniMax-M2.7"
    )


def build_openai_compatible_chat(model_name: str) -> ChatOpenAI:
    base = require_env(
        "OPENAI_BASE_URL",
        "使用 Minimax OpenAI 兼容接口时需设置（官方示例: https://api.minimaxi.com/v1）。",
    )
    key = require_env("OPENAI_API_KEY", "使用 OpenAI 兼容接口时需设置。")
    return ChatOpenAI(
        model=model_name,
        base_url=base.rstrip("/"),
        api_key=key,
    )


def minimax_model_name_from_spec(spec: str) -> str:
    name = spec.split(":", 1)[1].strip()
    return name or default_openai_compat_model()


def resolve_chat_model(cli_model: str | None) -> str | BaseChatModel | None:
    """返回 create_deep_agent 的 model：字符串、ChatOpenAI、MiniMaxChat 或 None（库默认 Anthropic）。"""
    spec = (cli_model or os.environ.get("DEEPAGENTS_MODEL", "").strip() or None)

    if spec and spec.lower().startswith("minimax:"):
        name = minimax_model_name_from_spec(spec)
        if os.environ.get("OPENAI_BASE_URL", "").strip() and os.environ.get(
            "OPENAI_API_KEY", ""
        ).strip():
            return build_openai_compatible_chat(name)
        require_env(
            "MINIMAX_API_KEY",
            "未配置 OpenAI 兼容接口时，使用 Minimax 原生接口需要 MINIMAX_API_KEY。",
        )
        return MiniMaxChat(model=name)

    if spec:
        return spec

    if os.environ.get("OPENAI_BASE_URL", "").strip() and os.environ.get(
        "OPENAI_API_KEY", ""
    ).strip():
        return build_openai_compatible_chat(default_openai_compat_model())

    if os.environ.get("MINIMAX_API_KEY", "").strip():
        return MiniMaxChat(
            model=os.environ.get("MINIMAX_MODEL", "abab6.5s-chat").strip()
        )

    return None


def ensure_provider_key(model: str | BaseChatModel | None) -> None:
    if isinstance(model, ChatOpenAI):
        require_env("OPENAI_API_KEY", "当前使用 OpenAI 兼容客户端（如 Minimax）。")
        require_env("OPENAI_BASE_URL", "当前使用 OpenAI 兼容客户端（如 Minimax）。")
        return
    if isinstance(model, MiniMaxChat):
        require_env("MINIMAX_API_KEY", "当前使用 Minimax 原生 HTTP 接口。")
        return
    if model is None:
        require_env(
            "ANTHROPIC_API_KEY",
            "未配置 OPENAI_BASE_URL+OPENAI_API_KEY 或 MINIMAX_API_KEY，默认使用 Anthropic。",
        )
        return
    prefix = model.split(":", 1)[0].strip().lower()
    if prefix == "ollama":
        return
    env_name = _PROVIDER_ENV.get(prefix)
    if env_name:
        require_env(env_name, f"当前模型为 {model}。")
