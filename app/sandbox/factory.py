"""沙箱后端工厂：state / local / docker。"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

from deepagents.backends import LocalShellBackend, StateBackend
from deepagents.backends.protocol import BackendProtocol, SandboxBackendProtocol

from app.config import DEFAULT_DOCKER_IMAGE, SANDBOX_LOCAL_DEFAULT_ROOT
from app.sandbox.docker_backend import DockerSandbox
from app.sandbox.docker_user_runtime import ensure_user_bound_container

SandboxMode = Literal["state", "local", "docker"]


def resolve_sandbox_mode(cli: str | None) -> SandboxMode:
    raw = (cli or os.environ.get("DEEPAGENTS_SANDBOX") or "state").strip().lower()
    if raw in ("state", "memory", "default"):
        return "state"
    if raw in ("local", "shell", "localshell"):
        return "local"
    if raw in ("docker", "container"):
        return "docker"
    print(
        f"错误: 无效的沙箱模式 {raw!r}，请使用 state、local 或 docker（或环境变量 DEEPAGENTS_SANDBOX）。",
        file=sys.stderr,
    )
    sys.exit(1)


def _docker_cli_ok() -> bool:
    try:
        r = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def build_backend(
    mode: SandboxMode,
    workspace: Path | None,
    *,
    docker_image: str | None = None,
    docker_network: str | None = None,
    docker_code_root: str | None = None,
    docker_user_id: str | None = None,
) -> BackendProtocol:
    if mode == "state":
        return StateBackend()
    if mode == "local":
        root = workspace if workspace is not None else SANDBOX_LOCAL_DEFAULT_ROOT
        root.mkdir(parents=True, exist_ok=True)
        return LocalShellBackend(
            root_dir=str(root),
            virtual_mode=True,
            inherit_env=True,
        )
    # docker：默认「复制进容器」；若设置用户 ID 则改为「只读 /project + 命名卷 /workspace」持久化
    if not _docker_cli_ok():
        print(
            "错误: 未检测到可用 Docker（`docker info` 失败或未安装）。"
            "请先启动 Docker 后再使用 --sandbox docker。",
            file=sys.stderr,
        )
        sys.exit(1)
    image = (
        (docker_image or "").strip()
        or os.environ.get("DOCKER_SANDBOX_IMAGE", "").strip()
        or DEFAULT_DOCKER_IMAGE
    )
    net = (docker_network or os.environ.get("DOCKER_SANDBOX_NETWORK", "").strip() or None)
    net = net if net else None

    user_raw = (docker_user_id or os.environ.get("DOCKER_SANDBOX_USER_ID", "").strip() or None)
    if user_raw:
        code_txt_u = (docker_code_root or "").strip() or os.environ.get(
            "DOCKER_SANDBOX_CODE_ROOT", ""
        ).strip()
        code_host_u = (
            Path(code_txt_u).resolve() if code_txt_u else SANDBOX_LOCAL_DEFAULT_ROOT.resolve()
        )
        try:
            cname = ensure_user_bound_container(
                user_raw_id=user_raw,
                code_readonly_host=code_host_u,
                image=image,
                network=net,
            )
        except (RuntimeError, NotADirectoryError) as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
        root_u = workspace if workspace is not None else SANDBOX_LOCAL_DEFAULT_ROOT
        root_u.mkdir(parents=True, exist_ok=True)
        return DockerSandbox.attach(
            cname,
            workspace_host=root_u,
            readonly_project_mounted=True,
        )

    code_txt = (docker_code_root or "").strip() or os.environ.get(
        "DOCKER_SANDBOX_CODE_ROOT", ""
    ).strip()
    code_host = Path(code_txt).resolve() if code_txt else SANDBOX_LOCAL_DEFAULT_ROOT.resolve()
    ph = Path(tempfile.gettempdir()) / "deepagent-docker-ephemeral"
    return DockerSandbox.start(
        workspace_host=ph,
        image=image,
        network=net,
        code_readonly_host=None,
        copy_code_from=code_host,
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
    if isinstance(backend, DockerSandbox) and (
        backend.readonly_project_mounted or backend.project_copy_in_workspace
    ):
        r3 = backend.execute("python3 -c 'import app; print(\"import_ok\")'")
        if r3.exit_code != 0 or "import_ok" not in (r3.output or ""):
            hint = "快照 /workspace/project" if backend.project_copy_in_workspace else "只读 /project"
            print(
                f"沙箱验证失败（{hint} 下 import app）: exit_code={r3.exit_code!r} output={r3.output!r}",
                file=sys.stderr,
            )
            return 1

    print(
        "沙箱验证通过: execute(echo) 与 execute(python3) 均成功。"
        f" 后端 id={backend.id!r}。"
    )
    return 0
