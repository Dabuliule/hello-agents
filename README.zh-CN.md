# CodeCraft

[English](README.md)

CodeCraft 是一个运行在本地代码仓库里的 Coding Agent Runtime。它关注的不是“再包一层模型接口”，而是把 Agent 真正执行任务时需要的运行时能力拆清楚：配置加载、提示词/项目指令、模型 provider、工具执行、审批、session 事件日志、恢复会话和诊断。

项目目前正在按 v1.0 runtime 方向重构。当前已经可以通过 Qwen、DeepSeek 或 OpenAI-compatible provider 运行真实的多轮 TUI 会话和可脚本化 CLI 任务。本地 bash 默认使用操作系统原生沙箱；Docker 仅作为可复现或更强隔离场景的显式选项。

许可证：Apache-2.0。

版本历史与升级说明见 [CHANGELOG.zh-CN.md](CHANGELOG.zh-CN.md)。

## CodeCraft 能做什么

- 通过 `exec` 执行可脚本化任务，通过 TUI 进行多轮交互和恢复会话。
- 从用户级、profile、项目级、显式配置文件和 CLI 参数加载配置。
- 注入内置 runtime instructions，并读取 workspace 内的 `AGENTS.md` / `CODECRAFT.md`。
- 调用 Qwen、DeepSeek 和 OpenAI 等 OpenAI-compatible provider。
- 通过结构化 tool schema 把工具暴露给模型，而不是把工具说明硬塞进 prompt。
- 所有工具调用都必须经过 `ToolRunner`。
- 对写文件、apply patch、bash 和高风险命令执行审批。
- bash 默认通过 macOS Seatbelt 或 Linux bubblewrap 隔离，也支持显式 process 和 Docker backend。
- 把 session 事件以 JSONL 存到 `~/.codecraft/sessions`。
- 从事件日志重建 conversation，用于 resume。
- 支持 CLI 查看和导出事件、工具调用、错误、原始日志和损坏 session。
- 内置 10 个固定 Coding Agent 任务，支持确定性判分和 JSON/HTML 成功率报告。
- 可连接 stdio MCP server，并让发现到的工具继续经过 sandbox、审批和 trace 链路。
- 可向其他 MCP host 提供只读仓库搜索和项目上下文。

## 安装

CodeCraft 目前适合作为 alpha / 试用版从 GitHub 安装。

使用 `uv` 安装：

```zsh
uv tool install git+https://github.com/Dabuliule/codecraft.git
```

使用 `uv` 更新：

```zsh
uv tool upgrade codecraft
```

使用 `pipx` 安装：

```zsh
pipx install git+https://github.com/Dabuliule/codecraft.git
```

使用 `pipx` 更新：

```zsh
pipx upgrade codecraft
```

安装后运行：

```zsh
codecraft
```

从源码本地开发：

```zsh
git clone https://github.com/Dabuliule/codecraft.git
cd codecraft
uv sync
uv run codecraft
```

## 当前 CLI

执行单轮任务：

```zsh
uv run codecraft exec "总结一下这个仓库"
```

启动全屏终端界面：

```zsh
uv run codecraft
```

TUI 使用以对话为中心的单列布局。工具调用在对话流中原位更新，精简后的 runtime 和 Token 状态显示在输入区下方。Assistant Markdown 在流式输出时原位更新。高风险工具调用会将输入区替换为内联选项，可以使用方向键选择操作并按回车确认。当前 turn 结束前输入框保持锁定。TUI 消费与 CLI 相同的 `RuntimeEvent`，没有实现第二套 Agent loop。

当前仓库存在历史 session 时，TUI 启动后会打开 session 浏览器，可以选择恢复原有配置和对话，也可以新建 session。还可以直接恢复：

```zsh
uv run codecraft --last
uv run codecraft --resume <session_id>
```

为了让长 session 的终端渲染保持流畅，恢复时只显示有限数量的历史消息；Runtime 仍会从事件日志重建全部可用模型上下文。

按下 `Ctrl+T` 可以直接在 TUI 中检查当前持久化 trace。Trace 界面复用标准报告模型，展示汇总指标、虚拟化事件表和结构化 payload。

列出有效 session：

```zsh
uv run codecraft sessions
```

列出有效和损坏 session：

```zsh
uv run codecraft sessions --all
```

检查 session：

```zsh
uv run codecraft inspect <session_id>
uv run codecraft inspect <session_id> --events
uv run codecraft inspect <session_id> --tools
uv run codecraft inspect <session_id> --errors
uv run codecraft inspect <session_id> --raw
```

`inspect --raw` 不做 JSONL 校验，适合排查已经损坏的 session log。

导出可读 trace 报告：

```zsh
uv run codecraft trace <session_id>
uv run codecraft trace <session_id> --format json
uv run codecraft trace <session_id> --output-dir ./traces
```

默认会同时写出 `<session_id>.trace.json` 和 `<session_id>.trace.html`。

运行内置 Coding Agent 评测：

```zsh
uv run codecraft eval --list
uv run codecraft eval
uv run codecraft eval --task locate-legacy-token
uv run codecraft eval --task locate-legacy-token --repeat 3
uv run codecraft eval --limit 3 --output-dir ./outputs/eval-run
```

任务覆盖文件创建、定点修改、多文件同步、仓库检索、结构化数据、重构、项目指令和受约束的数据清理。每次 attempt 都在单独生成的 workspace 中运行，并通过文件内容或 JSON 字段做确定性判分。报告包含每题成功率、p50/p95 时延、Token 用量、工具失败数和失败分类。输出目录会保留 `eval-report.json`、`eval-report.html`、各次 attempt 的 workspace 和 JSON trace。

评测任务只能读取、检索、写入和 patch 自己生成的 workspace，不提供 bash 工具，也不开网络。完整评测会产生真实模型 API 调用，可以先用 `--task` 或 `--limit` 做小规模 smoke run。

第一版真实 provider baseline 已记录在 [`docs/EVAL_BASELINE.md`](docs/EVAL_BASELINE.md)：Qwen3.7 Max Preview 完整运行一次 10 题，成功率为 70.0%，总 Token 为 200,485，p50 延迟为 13.4 秒，p95 延迟为 22.6 秒。精确 grader 暴露了 patch runtime 的尾换行缺陷；文档将原始 baseline 与修复后的定向重跑分开记录。自动化 CI 仍使用 mock provider，不消耗模型 API 额度。

## 配置

CodeCraft 使用 TOML 配置。优先级从高到低：

```text
CLI 显式参数 / --config
> 项目级 .codecraft/config.toml
> profile ~/.codecraft/profiles/<name>.toml
> 用户级 ~/.codecraft/config.toml
> 内置默认值
```

推荐的用户级配置：

```toml
# ~/.codecraft/config.toml
[model]
provider = "qwen"
name = "qwen-plus"
api_key_env = "DASHSCOPE_API_KEY"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
context_window_tokens = 131072
max_output_tokens = 8192

[approval]
policy = "on_request"

[sandbox]
mode = "workspace_write"
backend = "auto"
network_access = false
env_allowlist = []

[instructions]
user = "回答尽量简洁。"

[turn]
max_tool_calls = 30
max_tool_output_chars = 80000
max_tool_output_tokens = 16384
turn_timeout_seconds = 1800
tool_timeout_seconds = 300
approval_timeout_seconds = 300
context_safety_margin_tokens = 2048
context_keep_recent_items = 12
max_parallel_read_tools = 4
```

运行时会按配置的模型上下文窗口估算输入 token，并预留最大输出和安全余量。
较早的完整 turn 会自动压缩成不可信的历史摘要，当前工具协议保持完整；工具
结果还会根据下一次模型调用的剩余输入预算动态限额。

API key 建议放在环境变量里：

```zsh
export DASHSCOPE_API_KEY="你的 key"
```

CodeCraft 推荐使用 `api_key_env`，不推荐把明文 API key 写进 TOML。如果不配置 `api_key_env`，CodeCraft 会按 provider 使用默认值：Qwen 使用 `DASHSCOPE_API_KEY`，DeepSeek 使用 `DEEPSEEK_API_KEY`，OpenAI 使用 `OPENAI_API_KEY`。`base_url` 可选；Qwen 默认使用 DashScope compatible mode endpoint，DeepSeek 默认使用 `https://api.deepseek.com`。

常用 CLI 覆盖：

```zsh
uv run codecraft --provider qwen --model qwen-plus
uv run codecraft --provider deepseek --model deepseek-v4-flash
uv run codecraft --config ./my-config.toml
uv run codecraft --profile work
uv run codecraft --approval-policy on_request
uv run codecraft --network
```

## Provider

当前 provider：

- `qwen`
- `deepseek`
- `openai`

它们都基于 OpenAI-compatible provider。Qwen 和 DeepSeek 通过 Chat Completions 风格 adapter 实现，因为它们的兼容 API 提供 chat/function-style 调用形态。

内置默认模型配置：

```toml
[model]
provider = "qwen"
name = "qwen-plus"
# qwen 的 api_key_env 默认是 DASHSCOPE_API_KEY
# qwen 的 base_url 默认是 DashScope compatible mode
```

DeepSeek 示例：

```toml
[model]
provider = "deepseek"
name = "deepseek-v4-flash"
# deepseek 的 api_key_env 默认是 DEEPSEEK_API_KEY
# deepseek 的 base_url 默认是 https://api.deepseek.com
```

## Prompt 和 Instructions

每次调用模型前，CodeCraft 会组装一条 system message，来源包括：

```text
base_instructions
project_instructions
user_instructions
available_skills
active_skills
turn_context
conversation
```

项目指令会从 workspace 内安全加载：

```text
AGENTS.md
CODECRAFT.md
```

每个 turn 都会重新加载指令。根目录规则先应用；访问嵌套路径后会加载对应
目录作用域的规则，越深的作用域优先级越高。解析后指向 workspace 外部的
符号链接会被忽略。

工具 schema 不写进 prompt，而是通过 provider 的 `tools` 参数以结构化 schema 传给模型。

## Skills

Skill 是本地可复用的工作流指令，CodeCraft 通过渐进式加载控制上下文开销。
Runtime 启动时会从下面两个位置发现 Skill：

```text
<workspace>/.codecraft/skills/<name>/SKILL.md
~/.codecraft/skills/<name>/SKILL.md
```

第二个路径以配置的 `codecraft_home` 为准。同名时项目级 Skill 覆盖用户级
Skill，被覆盖来源会保留在 `SkillRegistry.diagnostics()` 中。

每个文件由严格的 YAML frontmatter 和 Markdown 指令正文组成：

```markdown
---
name: frontend-review
description: Use when reviewing frontend usability and accessibility.
---

# Frontend Review

修改前先检查键盘导航和焦点状态。
```

第一次请求只把 `name`、`description` 和 `source` 摘要放进 system prompt。
当某个 Skill 与当前任务匹配时，模型调用只读的 `load_skill` 工具；正文从
下一次模型请求开始进入 `<active_skills>`，并且只在当前 turn 内有效。

Skill 名称必须与目录一致，只能包含小写字母、数字、连字符或下划线；文件
必须是 UTF-8，最大 64 KiB，并且不能是符号链接。无效条目会被跳过并留下
诊断。CodeCraft 不会自动执行 Skill 中的脚本；Skill 提到的命令仍须使用普通
工具，并经过现有 sandbox 和审批。修改 Skill 文件后需要重启 runtime。

## 内置工具

| 工具 | 作用 | 说明 |
| --- | --- | --- |
| `read_file` | 读取 workspace 内文本文件 | 只读 |
| `list_files` | 列出 workspace 内文件/目录 | 跳过常见噪声目录 |
| `workspace_search` | 搜索 workspace 内路径和文本内容 | 返回路径、行号和片段 |
| `write_file` | 写入 workspace 内文本文件 | 需要审批 |
| `apply_patch` | 在 workspace 内应用 unified diff | 需要审批 |
| `bash` | 在 workspace 内运行 shell 命令 | command policy + 审批 + native/process/Docker backend |
| `load_skill` | 为当前 turn 激活已发现的 Skill | 只读；仅在存在有效 Skill 时注册 |

所有工具都通过 `ToolRunner` 执行。`ToolRegistry` 只负责注册和查找工具，不执行工具。

## 审批和安全边界

审批策略：

| 策略 | 行为 |
| --- | --- |
| `never` | 不询问审批 |
| `on_request` | 对声明需要审批的工具/命令询问 |
| `untrusted` | 对有副作用的工具询问 |

CLI 会展示审批细节，尤其是 bash 命令：

```text
• bash python -c 'print(1)'
approval required
tool     bash
risk     prompt
reason   unknown command requires approval
command  python -c 'print(1)'
Approve? [y/n/d]
```

Command policy 会区分安全命令、需要审批的命令、拒绝命令和网络命令。`python --version` 和 `python -V` 是安全命令；任意 Python 代码执行需要审批。

Sandbox mode：

| 模式 | 行为 |
| --- | --- |
| `read_only` | 只允许只读工具 |
| `workspace_write` | 允许 workspace 写入和进程执行，但仍受审批和 command policy 约束 |
| `danger_full_access` | 不再按 mode 限制 tool effect；网络策略、审批和 backend 隔离仍然生效 |

Sandbox mode 决定哪些能力可以执行，backend 决定审批通过的 bash 进程在哪里运行。默认 `auto` backend 会选择 macOS Seatbelt 或 Linux bubblewrap；没有受支持的 OS 沙箱时会关闭失败，不会悄悄退化成无隔离执行。

### 原生 Sandbox

默认 backend 不要求安装 Docker：

- macOS 使用系统内置的 `/usr/bin/sandbox-exec` Seatbelt runtime；
- Linux 使用 `bwrap`，需要通过系统包管理器安装；
- 不支持的平台会拒绝 bash，而不是无提示地在宿主机执行。

原生 backend 允许读取宿主机文件，在 `workspace_write` 下只允许写 workspace，为命令提供独立临时目录，并在 `network_access=false` 时从 OS 层拒绝网络系统调用；子进程继承同一边界。Bash 只会继承少量安全环境变量以及 `sandbox.env_allowlist` 显式列出的名称，模型 API Key 默认不会进入命令环境。

只有明确接受无 OS 隔离时才配置 `backend = "process"`。也可以显式选择 `seatbelt` 或 `bubblewrap` 做诊断。Docker 继续用于 CI、Eval、不可信仓库和可复现工具链。

### Docker Sandbox

先构建项目提供的镜像：

```zsh
docker build -f docker/sandbox.Dockerfile -t codecraft-sandbox:py311 docker
```

然后通过配置启用：

```toml
[sandbox]
mode = "workspace_write"
backend = "docker"
network_access = false
env_allowlist = []

[sandbox.docker]
image = "codecraft-sandbox:py311"
cpus = 1.0
memory_mb = 1024
pids_limit = 256
tmpfs_mb = 256
```

CodeCraft 使用 `--pull never` 启动容器，因此镜像必须预先存在于本机。项目需要其他语言或工具链时，可以配置自定义镜像。

Docker backend 为每条 bash 命令创建一个临时容器。容器根文件系统只读，`/tmp` 使用有容量限制的 tmpfs，并使用宿主机 UID/GID、移除 Linux capabilities、启用 `no-new-privileges`、限制 CPU/内存/PID、只挂载 workspace、通过 `sandbox.env_allowlist` 仅转发显式允许的环境变量；关闭网络时还会使用 `--network none`。超时容器会被强制清理。

重要边界：Docker 隔离的是 bash 进程，不会让挂载的仓库免受主动写入。在 `workspace_write` 模式下，真实 workspace 会以可写方式挂载；内置文件工具仍在宿主机运行并受 `WorkspaceGuard` 约束。审批和 command policy 仍是安全模型的一部分。

## MCP 工具

CodeCraft 可以启动显式配置的 stdio MCP server，完成 MCP lifecycle 握手和工具发现，并以 `mcp__project_tools__lookup` 这样的命名空间暴露给模型。每个 server 在一次 runtime 生命周期内复用一个连接，并在 runtime 退出时关闭。

```toml
[mcp.servers.project_tools]
command = "python"
args = ["tools/mcp_server.py"]
timeout_seconds = 30
max_tools = 128
env_allowlist = ["PROJECT_API_TOKEN"]

[mcp.servers.project_tools.tools.calculate]
effects = ["read_only"]
requires_approval = false

[mcp.servers.project_tools.tools.lookup]
effects = ["network", "external"]
requires_approval = true
```

远端 JSON Schema 同时用于模型可见的 tool schema 和本地参数校验。文本和 structured result 会被保留；二进制内容只记录摘要，避免把 base64 写进 session log。工具名会经过命名空间、字符清理和长度限制，以兼容模型 provider。

MCP tool annotations 只作为诊断 metadata 保存，绝不直接用于授权。没有 per-tool 配置覆盖时，发现到的工具默认带 `network` 和 `external` effect，并要求审批；当 `network_access=false` 时，这些默认调用会被 `SandboxPolicy` 拒绝。

重要边界：显式配置 stdio server，等于授权 CodeCraft 在工具发现前启动这个宿主机进程。子进程只继承 MCP SDK 的少量安全默认环境变量和 `env_allowlist` 中的变量，但当前不会自动进入 Docker bash sandbox。Network effect 可以约束工具调用，不能从物理上移除一个已受信任宿主进程的网络能力。

### 对外提供仓库上下文

CodeCraft 也可以作为只读 stdio MCP server，供其他 Agent host 使用：

```zsh
codecraft mcp-server --workspace /仓库的绝对路径
```

它会暴露结构化 `search_repository` 工具和两个 resources：

```text
codecraft://workspace/metadata
codecraft://workspace/instructions
```

搜索工具复用 `ContextEngine`，有索引时使用 lexical/symbol retrieval，并保留确定性 scan fallback。搜索路径必须经过 `WorkspaceGuard`；这个 server 不暴露 write、patch、bash 或 Agent loop。

CodeCraft client 可以用下面的只读本地策略连接：

```toml
[mcp.servers.codecraft_repo]
command = "codecraft"
args = ["mcp-server", "--workspace", "/仓库的绝对路径"]

[mcp.servers.codecraft_repo.tools.search_repository]
effects = ["read_only"]
requires_approval = false
```

## Session 和 Resume

Session 事件以 JSONL 存储：

```text
~/.codecraft/sessions/YYYY/MM/DD/<session_id>.jsonl
```

事件日志包含 session、turn、user、assistant、model tool call、tool start/finish、approval、token、error、finish 等事件。

事件与其中的 session 配置分别携带 schema 版本。缺失或未知版本会被明确拒绝，避免按不完整语义恢复。

`codecraft --last` 会加载当前 workspace 最近一个有效 session，从事件重建 conversation，然后继续对话，不会重新执行历史工具。

如果 session log 损坏，默认列表会跳过它。要查看损坏日志：

```zsh
uv run codecraft sessions --all
```

要排查损坏日志：

```zsh
uv run codecraft inspect <session_id> --raw
```

## 开发

从仓库内安装和运行：

```zsh
uv sync
uv run codecraft
```

质量检查：

```zsh
uv run ruff check .
uv run pytest
```

构建 sandbox 镜像后，可以显式运行 Docker 集成测试：

```zsh
CODECRAFT_RUN_DOCKER_TESTS=1 uv run pytest -m integration
```

普通测试会跳过这项集成测试；CI 会在独立 job 中构建镜像并真实运行。

Conventional Commits（提交信息不符合时会阻止 push）：

```zsh
git config core.hooksPath .githooks
chmod +x .githooks/pre-push
```

提交格式：

```text
type(scope): subject
```

示例：

```text
feat(tui): add session search
fix(tool): handle empty apply_patch payload
chore: update workflow permissions
```

当前测试覆盖 runtime events、session store、resume、配置加载、prompt 注入、模型 provider、MCP stdio client/server 互操作、tool runner、workspace 工具、bash policy、native/process/Docker sandbox backend、approval flow、CLI 行为和 Textual pilot 交互。

## 当前限制

- Docker 隔离依赖已经安装并运行的 Docker engine，以及预先构建的镜像。
- Docker 当前为每条命令创建临时容器，还没有 warm pool 或持久化工具缓存。
- Docker 只隔离 bash 执行；内置文件工具仍使用宿主机 workspace guard。
- MCP client 只消费 stdio tools；尚未消费 Streamable HTTP、resources、prompts 和动态 tool-list notification。
- CodeCraft MCP server 只暴露仓库搜索和两个只读 resources，不暴露通用 Agent 执行。
- 配置的 stdio MCP server 作为受信任宿主进程运行；尚未自动放进 Docker 隔离。
- TUI 同一时间只运行一个活动 session。
- 还没有自动清理 invalid session。
- v1.0 暂不做 Web/GitHub/cloud 工具。

## Runtime 结构

高层流程：

```text
CLI / TUI
  -> ConfigLoader
  -> AgentRuntime
  -> AgentThread / Session / Turn
  -> LLMProvider
  -> ToolRunner
  -> ApprovalManager
  -> SessionStore JSONL events
```

核心目录：

```text
src/codecraft/
  approval/      审批策略和 reviewer
  cli/           Typer CLI
  config/        TOML 配置模型和加载器
  core/          runtime、session、turn、事件重建
  eval/          固定 Agent 任务、确定性判分、评测报告
  llm/           provider 接口和 OpenAI-compatible provider
  prompt/        内置指令、项目指令加载、prompt builder
  sandbox/       命令和沙箱策略
  schema/        runtime、session、input、tool schema
  tool/          tool 抽象、registry、runner、内置工具
```
