"""项目级常量（路径、文档链接、默认镜像等）。"""

from pathlib import Path

DOCS_URL = "https://docs.langchain.com/oss/python/deepagents/quickstart"

# local 模式虚拟根：默认同仓库根
SANDBOX_LOCAL_DEFAULT_ROOT = Path(__file__).resolve().parent.parent

# Docker：容器内工作区与只读项目挂载点（与 DockerSandbox 内路径映射一致）
DOCKER_WORKSPACE_MOUNT = "/workspace"
# 只读挂载项目根后，容器内路径（与 PYTHONPATH 一致，便于 import app）
DOCKER_PROJECT_MOUNT = "/project"
DEFAULT_DOCKER_IMAGE = "python:3.12-slim"
DEFAULT_DOCKER_EXECUTE_TIMEOUT = 120
MAX_DOCKER_OUTPUT_BYTES = 100_000
# 复制进容器模式下 /workspace 使用 tmpfs 的上限（容器内内存盘，删容器即无）
DOCKER_WORKSPACE_TMPFS_SIZE = "2g"
