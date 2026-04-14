"""本地 Docker 容器作为 Deep Agents 沙箱：实现 SandboxBackendProtocol（继承 BaseSandbox）。"""

from __future__ import annotations

import atexit
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.sandbox import BaseSandbox

from app.config import (
    DEFAULT_DOCKER_EXECUTE_TIMEOUT,
    DOCKER_PROJECT_MOUNT,
    DOCKER_WORKSPACE_MOUNT,
    DOCKER_WORKSPACE_TMPFS_SIZE,
    MAX_DOCKER_OUTPUT_BYTES,
)

# 快照代码在容器可写卷内的目录（与 PYTHONPATH=/workspace/project 一致）
DOCKER_PROJECT_COPY_DIR = "project"


def _tar_copy_host_tree_to_workspace_project(container_id: str, source: Path) -> None:
    """将宿主目录树打包解压到容器 ``/workspace/project``（不挂载宿主代码目录）。"""
    source = source.resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"复制源码根不是目录: {source}")
    tar_cmd = [
        "tar",
        "-C",
        str(source),
        "-cf",
        "-",
        "--exclude=.git",
        "--exclude=.venv",
        "--exclude=.agent_docker_workspace",
        ".",
    ]
    p_tar = subprocess.Popen(tar_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p_tar.stdout is not None
    try:
        proc = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                container_id,
                "sh",
                "-c",
                f"mkdir -p {DOCKER_WORKSPACE_MOUNT}/{DOCKER_PROJECT_COPY_DIR} && "
                f"tar xf - -C {DOCKER_WORKSPACE_MOUNT}/{DOCKER_PROJECT_COPY_DIR}",
            ],
            stdin=p_tar.stdout,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    finally:
        p_tar.stdout.close()
        rc_tar = p_tar.wait(timeout=120)
    if rc_tar != 0:
        err = (p_tar.stderr.read() if p_tar.stderr else b"").decode(errors="replace")
        raise RuntimeError(f"打包源码失败 (tar exit {rc_tar}): {err[:800]}")
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"解压到容器 /workspace/{DOCKER_PROJECT_COPY_DIR} 失败: {msg[:800]}")


class DockerSandbox(BaseSandbox):
    """在本地 Docker 容器内执行命令与文件操作。

    本 demo 工厂路径：**快照复制**——不挂载宿主仓库；``/workspace`` 为容器 **tmpfs**，
    启动后将宿主代码树复制到 ``/workspace/project``（``PYTHONPATH=/workspace/project``），
    工具中的 ``/project/...`` 映射到该副本；预制代码树应视为只读，产出写在 ``/workspace`` 其他路径。
    ``docker rm`` 后数据即失。

    另提供 :meth:`attach` 连接已有容器（高级用法，CLI 未暴露）。
    """

    def __init__(
        self,
        *,
        container_id: str,
        workspace_host: Path,
        default_timeout: int = DEFAULT_DOCKER_EXECUTE_TIMEOUT,
        manage_lifecycle: bool = True,
        readonly_project_mounted: bool = False,
        project_copy_in_workspace: bool = False,
    ) -> None:
        self._container_id = container_id.strip()
        self._workspace_host = workspace_host.resolve()
        self._default_timeout = default_timeout
        self._manage_lifecycle = manage_lifecycle
        self._readonly_project_mounted = readonly_project_mounted
        self._project_copy_in_workspace = project_copy_in_workspace
        self._removed = False
        if manage_lifecycle:
            atexit.register(self._cleanup)

    @property
    def readonly_project_mounted(self) -> bool:
        return self._readonly_project_mounted

    @property
    def project_copy_in_workspace(self) -> bool:
        """为 True 时表示代码为启动时快照于 ``/workspace/project``，宿主仓库未挂载进容器。"""
        return self._project_copy_in_workspace

    @classmethod
    def start(
        cls,
        *,
        workspace_host: Path,
        image: str,
        network: str | None = None,
        extra_run_args: list[str] | None = None,
        code_readonly_host: Path | None = None,
        copy_code_from: Path | None = None,
    ) -> DockerSandbox:
        """``docker run`` 启动常驻容器（``sleep infinity``），退出进程时 ``docker rm -f``。

        Args:
            workspace_host: 非复制模式：绑定挂载到容器 ``/workspace`` 的宿主目录。复制模式：仅占位，不挂载。
            code_readonly_host: 若设置，额外以只读挂载到 ``/project``，并设置 ``PYTHONPATH=/project``。
            copy_code_from: 若设置，**不**挂载只读代码卷；``/workspace`` 使用容器 **tmpfs**；
                启动后将该宿主目录树复制到 ``/workspace/project``，并设置 ``PYTHONPATH=/workspace/project``。
        """
        if copy_code_from is not None and code_readonly_host is not None:
            raise ValueError("copy_code_from 与 code_readonly_host 不能同时设置")
        workspace_host = workspace_host.resolve()
        copy_mode = copy_code_from is not None
        if not copy_mode:
            workspace_host.mkdir(parents=True, exist_ok=True)
        name = f"deepagent-{uuid.uuid4().hex[:12]}"
        cmd: list[str] = [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
        ]
        ro_project = code_readonly_host is not None
        if ro_project:
            cr = code_readonly_host.resolve()
            cmd.extend(
                [
                    "-v",
                    f"{cr}:{DOCKER_PROJECT_MOUNT}:ro",
                    "-e",
                    "PYTHONPATH=/project",
                ]
            )
        elif copy_mode:
            cmd.extend(
                [
                    "-e",
                    f"PYTHONPATH={DOCKER_WORKSPACE_MOUNT}/{DOCKER_PROJECT_COPY_DIR}",
                ]
            )
        if copy_mode:
            cmd.extend(
                [
                    "--tmpfs",
                    f"{DOCKER_WORKSPACE_MOUNT}:rw,exec,nosuid,size={DOCKER_WORKSPACE_TMPFS_SIZE}",
                    "-w",
                    DOCKER_WORKSPACE_MOUNT,
                ]
            )
        else:
            cmd.extend(
                [
                    "-v",
                    f"{workspace_host}:{DOCKER_WORKSPACE_MOUNT}",
                    "-w",
                    DOCKER_WORKSPACE_MOUNT,
                ]
            )
        if network:
            cmd.extend(["--network", network])
        if extra_run_args:
            cmd.extend(extra_run_args)
        cmd.extend([image, "sleep", "infinity"])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"docker run 失败: {err}")
        cid = (proc.stdout or "").strip()
        if not cid and name:
            inspect = subprocess.run(
                ["docker", "inspect", "-f", "{{.Id}}", name],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if inspect.returncode == 0:
                cid = (inspect.stdout or "").strip()
        if not cid:
            raise RuntimeError("docker run 未返回容器 ID")
        if copy_mode:
            _tar_copy_host_tree_to_workspace_project(cid, copy_code_from.resolve())
        return cls(
            container_id=cid,
            workspace_host=workspace_host,
            default_timeout=DEFAULT_DOCKER_EXECUTE_TIMEOUT,
            manage_lifecycle=True,
            readonly_project_mounted=ro_project,
            project_copy_in_workspace=copy_mode,
        )

    @classmethod
    def attach(
        cls,
        container_id: str,
        *,
        workspace_host: Path,
        default_timeout: int = DEFAULT_DOCKER_EXECUTE_TIMEOUT,
        readonly_project_mounted: bool = False,
    ) -> DockerSandbox:
        """连接已有容器（挂载与路径约定需自行保证与 :meth:`start` 一致）。"""
        return cls(
            container_id=container_id,
            workspace_host=workspace_host,
            default_timeout=default_timeout,
            manage_lifecycle=False,
            readonly_project_mounted=readonly_project_mounted,
            project_copy_in_workspace=False,
        )

    def _cleanup(self) -> None:
        if not self._manage_lifecycle or self._removed:
            return
        self._removed = True
        subprocess.run(
            ["docker", "rm", "-f", self._container_id],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    @property
    def id(self) -> str:
        return self._container_id[:20]

    def _to_container_path(self, agent_path: str) -> str:
        p = agent_path if isinstance(agent_path, str) else str(agent_path)
        if p in (".", "..", ""):
            return DOCKER_WORKSPACE_MOUNT
        if p.startswith("/tmp/") or p.startswith("/var/tmp/"):
            return p
        if self._project_copy_in_workspace:
            if p == DOCKER_PROJECT_MOUNT or p.startswith(DOCKER_PROJECT_MOUNT + "/"):
                rel = p[len(DOCKER_PROJECT_MOUNT) :].lstrip("/")
                base = f"{DOCKER_WORKSPACE_MOUNT}/{DOCKER_PROJECT_COPY_DIR}"
                return f"{base}/{rel}" if rel else base
        elif p == DOCKER_PROJECT_MOUNT or p.startswith(DOCKER_PROJECT_MOUNT + "/"):
            return p
        if p == "/":
            return DOCKER_WORKSPACE_MOUNT
        if p == DOCKER_WORKSPACE_MOUNT or p.startswith(DOCKER_WORKSPACE_MOUNT + "/"):
            return p
        inner = p.lstrip("/")
        return f"{DOCKER_WORKSPACE_MOUNT}/{inner}" if inner else DOCKER_WORKSPACE_MOUNT

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        eff = timeout if timeout is not None else self._default_timeout
        if eff <= 0:
            eff = DEFAULT_DOCKER_EXECUTE_TIMEOUT
        try:
            proc = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-i",
                    "-w",
                    DOCKER_WORKSPACE_MOUNT,
                    self._container_id,
                    "sh",
                    "-c",
                    command,
                ],
                capture_output=True,
                text=True,
                timeout=float(eff) + 15.0,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                output=f"Error: docker exec 超时（{eff}s）",
                exit_code=124,
                truncated=False,
            )
        except FileNotFoundError:
            return ExecuteResponse(
                output="Error: 未找到 docker 命令，请安装 Docker 并加入 PATH。",
                exit_code=127,
                truncated=False,
            )
        except OSError as e:
            return ExecuteResponse(
                output=f"Error: 执行 docker 失败 ({type(e).__name__}): {e}",
                exit_code=1,
                truncated=False,
            )

        parts: list[str] = []
        if proc.stdout:
            parts.append(proc.stdout)
        if proc.stderr:
            parts.extend(f"[stderr] {ln}" for ln in proc.stderr.rstrip().split("\n"))
        output = "\n".join(parts) if parts else "<no output>"
        truncated = False
        if len(output) > MAX_DOCKER_OUTPUT_BYTES:
            output = (
                output[:MAX_DOCKER_OUTPUT_BYTES]
                + f"\n\n... 输出已截断（{MAX_DOCKER_OUTPUT_BYTES} 字节）"
            )
            truncated = True
        if proc.returncode != 0:
            output = f"{output.rstrip()}\n\nExit code: {proc.returncode}"
        return ExecuteResponse(output=output, exit_code=proc.returncode, truncated=truncated)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            cpath = self._to_container_path(path)
            tmp = tempfile.NamedTemporaryFile(delete=False)
            try:
                tmp.write(content)
                tmp.flush()
                tmp.close()
                proc = subprocess.run(
                    ["docker", "cp", tmp.name, f"{self._container_id}:{cpath}"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
            if proc.returncode != 0:
                msg = (proc.stderr or proc.stdout or "docker cp 失败").strip()
                responses.append(FileUploadResponse(path=path, error=msg[:500]))
            else:
                responses.append(FileUploadResponse(path=path, error=None))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(FileDownloadResponse(path=path, content=None, error="invalid_path"))
                continue
            cpath = self._to_container_path(path)
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp_path = tmp.name
            tmp.close()
            try:
                proc = subprocess.run(
                    ["docker", "cp", f"{self._container_id}:{cpath}", tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                if proc.returncode != 0:
                    msg = (proc.stderr or proc.stdout or "").lower()
                    err = (
                        "file_not_found"
                        if "no such file" in msg or "could not find" in msg
                        else "permission_denied"
                    )
                    responses.append(FileDownloadResponse(path=path, content=None, error=err))
                    continue
                data = Path(tmp_path).read_bytes()
                responses.append(FileDownloadResponse(path=path, content=data, error=None))
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        return responses

    def ls(self, path: str) -> LsResult:
        return super().ls(self._to_container_path(path))

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return super().read(self._to_container_path(file_path), offset=offset, limit=limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        return super().write(self._to_container_path(file_path), content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return super().edit(
            self._to_container_path(file_path),
            old_string,
            new_string,
            replace_all=replace_all,
        )

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        raw = path or "."
        mapped = DOCKER_WORKSPACE_MOUNT if raw in (".", "..") else self._to_container_path(raw)
        return super().grep(pattern, mapped, glob)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        mapped = self._to_container_path(path)
        return super().glob(pattern, mapped)
