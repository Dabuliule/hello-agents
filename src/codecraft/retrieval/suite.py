from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RETRIEVAL_SUITE_NAME = "codecraft-repository-retrieval-v1"


@dataclass(frozen=True)
class RetrievalCase:
    """固定语料上的查询及路径级相关性标注。

    ``relevant_paths`` 相当于检索评测里的 qrels：它声明哪些文件与查询相关，
    不评价同一文件内某个 snippet 的行号或文本质量。``path`` 和 ``mode`` 则
    进入真实 ``workspace_search`` 参数，用于覆盖目录作用域及路径/内容检索。
    dataclass 冻结后，一次 run 内的 repeat 都共享同一份不可变评测契约。
    """

    case_id: str
    category: str
    query: str
    relevant_paths: tuple[str, ...]
    path: str = "."
    mode: Literal["both", "content", "path"] = "both"


_CORPUS = {
    "config/features.py": "FEATURE_FLAG_EXPORT = True\nFEATURE_FLAG_AUDIT = False\n",
    "config/settings.toml": "payment_timeout_ms = 2500\nqueue_workers = 4\n",
    "docs/payments.md": (
        "# Payments\n\nClients must send an idempotency key when creating a charge.\n"
    ),
    "src/api/request.py": (
        "def attach_request_context(request, trace_id):\n"
        "    request.state.trace_id = trace_id\n"
    ),
    "src/auth/permissions.py": (
        "def authorize(actor, action):\n"
        "    allowed_actions = ROLE_PERMISSIONS.get(actor.role, set())\n"
        "    return action in allowed_actions\n"
    ),
    "src/auth/service.py": (
        "def validate_access_token(token):\n"
        "    claims = decode_and_verify(token)\n"
        "    return claims.subject\n"
    ),
    "src/billing/invoice.py": (
        "class InvoiceBuilder:\n"
        "    def create(self, order):\n"
        "        return Invoice(order_id=order.id)\n"
    ),
    "src/db/pool.go": (
        "package db\n\n"
        "const maxReconnectAttempts = 5\n"
        "func openPool(dsn string) *Pool { return connectWithBackoff(dsn) }\n"
    ),
    "src/observability/logging.py": (
        "def bind_trace(logger, trace_id):\n    return logger.bind(trace_id=trace_id)\n"
    ),
    "src/queue/worker.ts": (
        "export function processJob(job: Job) {\n"
        "  if (job.attempts > 4) throw new Error('retry budget exhausted');\n"
        "}\n"
    ),
    "src/services/payment_gateway.ts": (
        "export class PaymentGateway {\n"
        "  async charge(request: ChargeRequest) { return this.client.send(request); }\n"
        "}\n"
    ),
    "tests/test_permissions.py": (
        "def test_viewer_cannot_delete():\n"
        "    assert authorize(viewer, 'delete') is False\n"
    ),
}

_CASES = (
    RetrievalCase(
        "exact-symbol",
        "exact",
        "validate_access_token",
        ("src/auth/service.py",),
        mode="content",
    ),
    RetrievalCase(
        "exact-error-message",
        "exact",
        "retry budget exhausted",
        ("src/queue/worker.ts",),
        mode="content",
    ),
    RetrievalCase(
        "config-key",
        "exact",
        "payment_timeout_ms",
        ("config/settings.toml",),
        mode="content",
    ),
    RetrievalCase(
        "path-fragment",
        "path",
        "invoice",
        ("src/billing/invoice.py",),
        mode="path",
    ),
    RetrievalCase(
        "multi-file-identifier",
        "multi_file",
        "trace_id",
        ("src/api/request.py", "src/observability/logging.py"),
        mode="content",
    ),
    RetrievalCase(
        "case-insensitive-symbol",
        "exact",
        "feature_flag_export",
        ("config/features.py",),
        mode="content",
    ),
    RetrievalCase(
        "scoped-doc-search",
        "scoped",
        "idempotency key",
        ("docs/payments.md",),
        path="docs",
        mode="content",
    ),
    RetrievalCase(
        "natural-language-permissions",
        "semantic",
        "where are user permissions checked",
        ("src/auth/permissions.py",),
        mode="content",
    ),
    RetrievalCase(
        "natural-language-reconnect",
        "semantic",
        "database connection retry policy",
        ("src/db/pool.go",),
        mode="content",
    ),
    RetrievalCase(
        "cross-language-type",
        "exact",
        "PaymentGateway",
        ("src/services/payment_gateway.ts",),
        mode="content",
    ),
)


def get_retrieval_cases() -> tuple[RetrievalCase, ...]:
    """返回用于横向比较检索实现的不可变、稳定用例集合。"""
    return _CASES


def seed_retrieval_workspace(workspace: Path) -> None:
    """创建评测使用的固定 Python/TS/Go/配置/文档语料和忽略项。

    语料完全由代码生成，不依赖调用者当前仓库，因此不同机器和不同检索策略
    面对的是同一组字节。benchmark 只读该 workspace，repeat 可以安全共享它；
    这和会修改文件、必须为每次 attempt 重新 seed 的 Agent Eval 不同。

    Example:
        ``seed_retrieval_workspace(tmp_path / "workspace")`` 会创建
        ``src/auth/service.py`` 等文件，以及不应被检索的 ``__pycache__`` 文件。
    """
    workspace.mkdir(parents=True, exist_ok=True)
    for relative_path, content in _CORPUS.items():
        target = workspace / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    ignored = workspace / "__pycache__" / "shadow.py"
    ignored.parent.mkdir(parents=True, exist_ok=True)
    ignored.write_text("validate_access_token = 'ignored'\n", encoding="utf-8")
