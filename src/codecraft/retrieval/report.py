from __future__ import annotations

import json
from html import escape
from typing import Any


def render_retrieval_json(report: dict[str, Any]) -> str:
    """将报告渲染为保留 Unicode、带缩进且以换行结束的 JSON。"""
    return json.dumps(report, ensure_ascii=False, indent=2) + "\n"


def render_retrieval_html(report: dict[str, Any]) -> str:
    """把报告摘要渲染成无外部资源的可分享 HTML 页面。

    所有来自 report 的字符串进入 HTML 前均经过 escape，避免查询、用例名称
    或指标内容成为可执行标记。HTML 展示 case 聚合结果；JSON 仍保留每一次
    repeat 的完整路由、候选成本和排名明细，排障时应以 JSON 为准。
    """
    run = report["run"]
    metrics = report["metrics"]
    rows = "\n".join(_case_row(case) for case in report["cases"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CodeCraft Retrieval Eval</title>
  <style>
    :root {{ color-scheme: light; --bg: #f6f8fb; --panel: #fff; --ink: #152033;
      --muted: #657187; --line: #d8dfeb; --accent: #116466; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 24px 48px; }}
    h1 {{ margin: 0; font-size: 28px; }}
    h2 {{ margin: 28px 0 10px; font-size: 18px; }}
    .muted {{ color: var(--muted); }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px; margin: 20px 0 28px; }}
    .metric {{ padding: 14px; background: var(--panel); border: 1px solid var(--line);
      border-radius: 8px; }}
    .metric strong {{ display: block; color: var(--accent); font-size: 24px; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel);
      border: 1px solid var(--line); }}
    th, td {{ padding: 10px; border-bottom: 1px solid var(--line); text-align: left;
      vertical-align: top; font-size: 13px; }}
    th {{ color: var(--muted); }}
    tr:last-child td {{ border-bottom: 0; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
  </style>
</head>
<body>
<main>
  <h1>CodeCraft Retrieval Eval</h1>
  <p class="muted">{escape(run["suite"])} · {escape(run["retriever"])} · {run["repeat"]} repeat(s)</p>
  <section class="metrics">
    {_metric("Recall@1", _score(metrics["mean_recall_at_1"]))}
    {_metric("Recall@5", _score(metrics["mean_recall_at_5"]))}
    {_metric("Precision@5", _score(metrics["mean_precision_at_5"]))}
    {_metric("MRR", _score(metrics["mean_reciprocal_rank"]))}
    {_metric("p50 Latency", _milliseconds(metrics["latency_p50_ms"]))}
    {_metric("p95 Latency", _milliseconds(metrics["latency_p95_ms"]))}
    {_metric("Mean Scanned Files", metrics["mean_scanned_files"])}
    {_metric("Mean Returned Tokens", metrics["mean_estimated_returned_tokens"])}
    {_metric("Zero Results", metrics["zero_result_count"])}
    {_metric("Irrelevant Paths", metrics["irrelevant_path_count"])}
  </section>
  <h2>Cases</h2>
  <table>
    <thead><tr><th>Case</th><th>Category</th><th>Query</th><th>Recall@1</th><th>Recall@5</th><th>Precision@5</th><th>MRR</th><th>p50</th><th>p95</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</main>
</body>
</html>
"""


def _metric(label: str, value: Any) -> str:
    """渲染一个转义后的指标卡。"""
    return (
        f'<div class="metric"><strong>{escape(str(value))}</strong>'
        f'<span class="muted">{escape(label)}</span></div>'
    )


def _case_row(case: dict[str, Any]) -> str:
    """渲染单个用例的转义 HTML 表格行。"""
    return (
        "<tr>"
        f"<td><code>{escape(case['case_id'])}</code></td>"
        f"<td>{escape(case['category'])}</td>"
        f"<td>{escape(case['query'])}</td>"
        f"<td>{_score(case['mean_recall_at_1'])}</td>"
        f"<td>{_score(case['mean_recall_at_5'])}</td>"
        f"<td>{_score(case['mean_precision_at_5'])}</td>"
        f"<td>{_score(case['mean_reciprocal_rank'])}</td>"
        f"<td>{escape(str(case['latency_p50_ms']))} ms</td>"
        f"<td>{escape(str(case['latency_p95_ms']))} ms</td>"
        "</tr>"
    )


def _score(value: float) -> str:
    """把检索质量分数固定格式化为三位小数。"""
    return f"{value:.3f}"


def _milliseconds(value: float) -> str:
    """为延迟数值添加毫秒单位。"""
    return f"{value} ms"
