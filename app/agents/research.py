"""研究型 Agent 的系统提示与组装入口。"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.memory import InMemorySaver

from deepagents import create_deep_agent

from app.sandbox.docker_backend import DockerSandbox
from app.tools.tavily import build_internet_search_tool

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol


RESEARCH_INSTRUCTIONS = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""


_DOCKER_PATH_HINT = """

## Docker sandbox paths (`execute` vs filesystem tools)

- `execute` runs in a container with working directory `/workspace`.
- Logical paths from file tools (for example `/home/user/script.py`) live **inside** the workspace as
  `/workspace/home/user/script.py`, not as bare `/home/user/...` on the filesystem.
- In shell commands, use `/workspace/...`, or paths relative to `/workspace` (for example `python home/user/script.py`).
  Do not `cd /home/user` or `python /home/user/...` unless that directory exists at the container root (it usually does not).
"""

_DOCKER_COPY_SNAPSHOT_HINT = """

## Docker code snapshot (copy mode)

- Project code was copied at container start into `/workspace/project` (not bind-mounted from the host).
- Treat that tree as **read-only**: do not edit files under `/project/...` or `/workspace/project/...`; write outputs elsewhere under `/workspace` (for example `/workspace/output/...`).
- `/workspace` is a container **tmpfs**; nothing is written under your host project tree for workspace files.
- Tool paths like `/project/app/...` resolve to that snapshot under `/workspace/project/...`.
- Removing the container discards all workspace and snapshot data.
"""


def create_research_agent(
    *,
    backend: "BackendProtocol",
    model: Any,
) -> Any:
    """创建带 Tavily 与给定后端的 deep agent。"""
    internet_search = build_internet_search_tool(os.environ["TAVILY_API_KEY"])
    checkpointer = InMemorySaver()
    system_prompt = RESEARCH_INSTRUCTIONS
    if isinstance(backend, DockerSandbox):
        system_prompt = RESEARCH_INSTRUCTIONS + _DOCKER_PATH_HINT
        if backend.project_copy_in_workspace:
            system_prompt += _DOCKER_COPY_SNAPSHOT_HINT
    return create_deep_agent(
        model=model,
        tools=[internet_search],
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        backend=backend,
    )
