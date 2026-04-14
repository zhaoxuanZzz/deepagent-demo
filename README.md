# deepagent-demo

基于 [LangChain Deep Agents](https://docs.langchain.com/oss/python/deepagents/quickstart) 的示例项目：研究型 Agent + **Tavily 联网搜索**，支持 **state / local / docker** 三种后端，控制台流式输出与工具调用日志（带去重）。

## 环境要求

- Python **3.12+**
- 推荐使用 [**uv**](https://docs.astral.sh/uv/) 管理依赖与虚拟环境
- **Tavily** 与模型 API：见下方环境变量
- **`--sandbox docker`**：本机已安装并启动 **Docker**（`docker info` 可用）；**不需要** docker-compose

## 快速开始

```bash
git clone <本仓库地址> && cd deepagent-demo
cp .env.example .env
# 编辑 .env：至少填写 TAVILY_API_KEY 与模型相关变量（见 .env.example 注释）

uv sync
uv run python main.py --query "搜索武汉最近的新闻"
```

多轮对话：

```bash
uv run python main.py --chat
```

查看全部参数：

```bash
uv run python main.py --help
```

## 环境变量

复制 `.env.example` 为 `.env` 后按需填写。常用项：

| 变量 | 说明 |
|------|------|
| `TAVILY_API_KEY` | Tavily 搜索（运行 Agent 时必填） |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` | OpenAI 兼容接口（如 Minimax 官方地址） |
| `MINIMAX_MODEL` / `OPENAI_MODEL` | 模型名 |
| `DEEPAGENTS_MODEL` | 显式指定厂商模型，如 `openai:gpt-4o`、`minimax:模型名` |
| `DEEPAGENTS_SANDBOX` | `state`（默认）\|`local`\|`docker` |
| `DOCKER_SANDBOX_IMAGE` | docker 模式镜像，默认 `python:3.12-slim` |
| `DOCKER_SANDBOX_NETWORK` | 传给 `docker run` 的 `--network`（如 `none`） |
| `DOCKER_SANDBOX_CODE_ROOT` | 默认 docker：复制源 → `/workspace/project`。用户级 docker：只读绑定到容器 `/project` 的宿主目录 |
| `DOCKER_SANDBOX_USER_ID` | 用户级 docker：绑定 `deepagent-u-*` 容器与 `deepagent-data-*` 命名卷；与「仅复制进容器」互斥 |

未配置 OpenAI 兼容或 Minimax 时，Deep Agents 默认使用 **Anthropic**，需 `ANTHROPIC_API_KEY`。

## 沙箱后端说明

| 模式 | 行为 | `execute` 工具 |
|------|------|----------------|
| **state** | 文件在 LangGraph 状态内，随 checkpoint / `thread_id` 持久策略变化 | 不提供 |
| **local** | `LocalShellBackend`：本机真实目录 + 本机 shell，**无隔离** | 提供（仅可信环境） |
| **docker** | **默认**：宿主目录打包复制到 `/workspace/project`，`/workspace` 为 **tmpfs**，删容器即无。**用户级**（`DOCKER_SANDBOX_USER_ID` / `--docker-user-id`）：宿主代码**只读**挂 `/project`，可写区为 Docker **命名卷**挂 `/workspace`，同一用户复用同一容器与卷 | 提供 |

**Docker 路径约定：**

- **默认（复制）**：`read_file("/project/app/...")` 映射到容器内快照 `/workspace/project/...`；产出写在 `/workspace` 下其他路径；`--sandbox-workspace` 不使用。
- **用户级**：读源码用容器内 `/project/...`（宿主只读挂载）；可写在 `/workspace`（命名卷持久化，进程退出**不**删容器）。清理示例：`docker rm -f deepagent-u-<slug>`；`docker volume rm deepagent-data-<slug>`（先删容器再删卷）。

```bash
# 默认 state
uv run python main.py --query "你好"

# 本机 shell（高风险）
uv run python main.py --sandbox local --chat

# Docker（自动 docker run 常驻容器，进程退出时清理）
uv run python main.py --sandbox docker --verify-sandbox
uv run python main.py --sandbox docker --chat

# Docker：按用户 ID 使用命名数据卷 /workspace（与「仅复制」互斥）
uv run python main.py --sandbox docker --docker-user-id alice --verify-sandbox
```

**工作区目录：**
- **local**：默认同项目根；`--sandbox-workspace` 为虚拟根。
- **docker（默认复制）**：`--docker-code-root` 为复制源；不使用宿主工作区挂载。
- **docker（用户级）**：`--docker-code-root` 为只读 `/project` 的宿主根；`--sandbox-workspace` 为占位宿主路径（工具映射用，可写数据在容器卷内）。

**仅验证沙箱**（不调用模型、不需要 Tavily）：

```bash
uv run python main.py --verify-sandbox
uv run python main.py --sandbox docker --verify-sandbox
```

## 项目结构

```
├── main.py                 # CLI 入口
├── app/
│   ├── config.py           # 常量与默认路径
│   ├── agents/             # Agent 组装与系统提示
│   ├── tools/              # Tavily 等工具
│   ├── sandbox/            # state/local/docker 工厂与 DockerSandbox
│   ├── models/             # 聊天模型解析与 Key 检查
│   ├── streaming/          # 控制台流式输出与工具日志
│   └── cli/                # argparse 与 main()
├── pyproject.toml
├── .env.example
└── quick_sort_verify.py    # 独立脚本示例（若有）
```

## 参考文档

- [Deep Agents Quickstart](https://docs.langchain.com/oss/python/deepagents/quickstart)
- [Deep Agents Sandboxes](https://docs.langchain.com/oss/python/deepagents/sandboxes)

## 许可证

以仓库内实际许可证文件为准（若未添加则默认仅作内部/学习用途）。
