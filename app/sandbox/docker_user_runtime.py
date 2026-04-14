"""按用户 ID 绑定 Docker 容器与命名卷：只读 ``/project`` + 命名卷 ``/workspace``。"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from app.config import DOCKER_PROJECT_MOUNT, DOCKER_WORKSPACE_MOUNT


def _run(cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _slug(user_raw_id: str) -> str:
    s = user_raw_id.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s).strip("-_.") or "user"
    return s[:48]


def _ensure_volume(volume_name: str) -> None:
    ins = _run(["docker", "volume", "inspect", volume_name], timeout=30)
    if ins.returncode == 0:
        return
    cr = _run(["docker", "volume", "create", volume_name], timeout=60)
    if cr.returncode != 0:
        err = (cr.stderr or cr.stdout or "").strip()
        raise RuntimeError(f"docker volume create {volume_name!r} 失败: {err}")


def _mounts_json(name: str) -> list[dict] | None:
    proc = _run(
        ["docker", "inspect", "-f", "{{json .Mounts}}", name],
        timeout=30,
    )
    if proc.returncode != 0:
        return None
    raw = (proc.stdout or "").strip()
    if not raw:
        return None
    try:
        out = json.loads(raw)
        return out if isinstance(out, list) else None
    except json.JSONDecodeError:
        return None


def _validate_user_mounts(
    mounts: list[dict],
    *,
    code_host: Path,
    volume_name: str,
) -> str | None:
    """挂载正确返回 ``None``，否则返回错误说明。"""
    code_s = str(code_host.resolve())
    proj_ok = False
    ws_ok = False
    for m in mounts:
        dest = str(m.get("Destination") or "")
        typ = str(m.get("Type") or "")
        src = str(m.get("Source") or "")
        rw = bool(m.get("RW", True))
        vol_name = str(m.get("Name") or "")
        if dest == DOCKER_PROJECT_MOUNT:
            if typ == "bind" and src == code_s and not rw:
                proj_ok = True
        if dest == DOCKER_WORKSPACE_MOUNT:
            if typ == "volume" and rw and (
                vol_name == volume_name or vol_name.endswith(f"/{volume_name}")
            ):
                ws_ok = True
    if not proj_ok or not ws_ok:
        return (
            f"挂载与预期不符（需宿主 {code_s!r} -> {DOCKER_PROJECT_MOUNT}:ro，"
            f"命名卷 {volume_name!r} -> {DOCKER_WORKSPACE_MOUNT}）。"
        )
    return None


def ensure_user_bound_container(
    *,
    user_raw_id: str,
    code_readonly_host: Path,
    image: str,
    network: str | None = None,
) -> str:
    """保证存在可 ``attach`` 的容器：创建卷、必要时 ``docker run`` / ``docker start``。

    返回容器名（可与 ``DockerSandbox.attach`` 联用）。
    """
    code_host = code_readonly_host.resolve()
    if not code_host.is_dir():
        raise NotADirectoryError(f"代码根不是目录: {code_host}")

    slug = _slug(user_raw_id)
    cname = f"deepagent-u-{slug}"
    vname = f"deepagent-data-{slug}"

    _ensure_volume(vname)

    ex = _run(["docker", "inspect", "-f", "{{.State.Running}}", cname], timeout=30)
    if ex.returncode == 0:
        mounts = _mounts_json(cname)
        if mounts is None:
            raise RuntimeError(f"无法读取容器 {cname!r} 的挂载信息。")
        bad = _validate_user_mounts(mounts, code_host=code_host, volume_name=vname)
        if bad:
            raise RuntimeError(f"{bad}请执行: docker rm -f {cname} 后重试。")
        running = (ex.stdout or "").strip().lower() == "true"
        if running:
            return cname
        st = _run(["docker", "start", cname], timeout=120)
        if st.returncode != 0:
            err = (st.stderr or st.stdout or "").strip()
            raise RuntimeError(f"docker start {cname!r} 失败: {err}")
        return cname

    cmd: list[str] = [
        "docker",
        "run",
        "-d",
        "--name",
        cname,
        "-v",
        f"{code_host}:{DOCKER_PROJECT_MOUNT}:ro",
        "-e",
        "PYTHONPATH=/project",
        "-v",
        f"{vname}:{DOCKER_WORKSPACE_MOUNT}",
        "-w",
        DOCKER_WORKSPACE_MOUNT,
    ]
    if network:
        cmd.extend(["--network", network])
    cmd.extend([image, "sleep", "infinity"])
    run_p = _run(cmd, timeout=120)
    if run_p.returncode != 0:
        err = (run_p.stderr or run_p.stdout or "").strip()
        raise RuntimeError(f"docker run 失败: {err}")

    mounts2 = _mounts_json(cname)
    if mounts2:
        bad2 = _validate_user_mounts(mounts2, code_host=code_host, volume_name=vname)
        if bad2:
            raise RuntimeError(f"{bad2}若挂载应一致，可尝试: docker start {cname}")

    return cname
