"""命令行参数定义。"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
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
        choices=["state", "local", "docker"],
        help="后端：state=图状态文件；local=本机 shell（高风险）；docker=本地容器（需 Docker）",
    )
    p.add_argument(
        "--sandbox-workspace",
        default=None,
        metavar="DIR",
        help=(
            "local：虚拟根目录。"
            "docker（默认复制进容器）：不使用。"
            "docker + --docker-user-id：占位宿主路径（供工具映射；可写数据在容器命名卷 /workspace）。"
        ),
    )
    p.add_argument(
        "--docker-image",
        default=None,
        metavar="IMAGE",
        help="--sandbox docker 时的镜像（默认 python:3.12-slim 或环境变量 DOCKER_SANDBOX_IMAGE）",
    )
    p.add_argument(
        "--docker-user-id",
        default=None,
        metavar="USER",
        help=(
            "docker：按用户绑定固定容器名 deepagent-u-* 与命名卷 deepagent-data-* 挂到 /workspace；"
            "与默认「复制进容器」互斥（设置后不再打包复制，改为只读挂载宿主代码到 /project）"
        ),
    )
    p.add_argument(
        "--docker-network",
        default=None,
        metavar="MODE",
        help="传给 docker run 的 --network，例如 none 以禁用容器网络",
    )
    p.add_argument(
        "--docker-code-root",
        default=None,
        metavar="DIR",
        help="docker：复制进容器的宿主源码根目录（默认仓库根），目标为容器内 /workspace/project",
    )
    p.add_argument(
        "--verify-sandbox",
        action="store_true",
        help="仅验证沙箱 execute，不调用模型（无需 TAVILY_API_KEY）",
    )
    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()
