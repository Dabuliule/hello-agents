from __future__ import annotations

import json
from datetime import datetime
from html import escape
from typing import Any

from codecraft.schema.event import RuntimeEvent, RuntimeEventType

TRACE_SCHEMA_VERSION = 1


def build_trace_report(session_id: str, events: list[RuntimeEvent]) -> dict[str, Any]:
    """Build a stable trace report from a session event log."""
    config = _session_config(events)
    first = events[0] if events else None
    last = events[-1] if events else None
    started_at = first.timestamp if first else None
    finished_at = last.timestamp if last else None

    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "session": {
            "session_id": session_id,
            "source": config.get("source"),
            "cwd": config.get("cwd"),
            "model": config.get("model"),
            "model_provider": config.get("model_provider"),
            "started_at": _iso(started_at),
            "finished_at": _iso(finished_at),
            "duration_ms": _duration_ms(started_at, finished_at),
        },
        "metrics": _metrics(events),
        "turns": _turns(events),
        "tool_calls": _tool_calls(events),
        "events": [_event_row(event) for event in events],
    }


def render_trace_json(report: dict[str, Any]) -> str:
    """渲染保留 Unicode、带缩进且以换行结束的稳定 Trace JSON。"""
    return json.dumps(report, ensure_ascii=False, indent=2) + "\n"


def render_trace_html(report: dict[str, Any]) -> str:
    """把 Trace 报告渲染为无外部依赖且所有动态文本均转义的 HTML。"""
    session = report["session"]
    metrics = report["metrics"]
    title = f"CodeCraft Trace {session['session_id']}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --panel: #ffffff;
      --ink: #0f172a;
      --muted: #64748b;
      --line: #dbe3ef;
      --accent: #0f766e;
      --error: #b91c1c;
      --ok: #166534;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1, h2 {{ margin: 0; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 18px; margin-top: 28px; }}
    .muted {{ color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin: 20px 0 8px;
    }}
    .metric, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .metric strong {{
      display: block;
      font-size: 24px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      margin-top: 12px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    tr:last-child td {{ border-bottom: 0; }}
    code, pre {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
    }}
    pre {{
      max-width: 520px;
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .status-ok {{ color: var(--ok); }}
    .status-failed, .status-error {{ color: var(--error); }}
    .pill {{
      display: inline-block;
      padding: 2px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #f1f5f9;
    }}
  </style>
</head>
<body>
<main>
  <h1>{escape(title)}</h1>
  <p class="muted">{escape(str(session.get("cwd") or ""))}</p>
  {_render_session_panel(session)}
  {_render_metrics(metrics)}
  {_render_tool_table(report["tool_calls"])}
  {_render_turn_table(report["turns"])}
  {_render_event_table(report["events"])}
</main>
</body>
</html>
"""


def _render_session_panel(session: dict[str, Any]) -> str:
    """渲染 Session 身份、来源、模型和时间范围面板。"""
    rows = [
        ("Session", session.get("session_id")),
        ("Source", session.get("source") or "-"),
        ("Model", _join_model(session)),
        ("Started", session.get("started_at") or "-"),
        ("Finished", session.get("finished_at") or "-"),
        ("Duration", _format_ms(session.get("duration_ms"))),
    ]
    items = "\n".join(
        f'<div><span class="muted">{escape(label)}</span><br>{escape(str(value))}</div>'
        for label, value in rows
    )
    return f'<section class="panel"><div class="grid">{items}</div></section>'


def _render_metrics(metrics: dict[str, Any]) -> str:
    """渲染事件、Turn、Tool、审批和错误计数卡片。"""
    cards = [
        ("Events", metrics["event_count"]),
        ("Turns", metrics["turn_count"]),
        ("Tool Calls", metrics["tool_call_count"]),
        ("Tool Failures", metrics["tool_failure_count"]),
        ("Approvals", metrics["approval_count"]),
        ("Errors", metrics["error_count"]),
    ]
    content = "\n".join(
        f'<div class="metric"><strong>{escape(str(value))}</strong><span class="muted">{escape(label)}</span></div>'
        for label, value in cards
    )
    return f'<h2>Metrics</h2><div class="grid">{content}</div>'


def _render_tool_table(tool_calls: list[dict[str, Any]]) -> str:
    """渲染工具终态、耗时和结果预览；空列表给出显式占位行。"""
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(call.get('seq') or '-'))}</td>"
        f"<td>{escape(str(call.get('name') or '-'))}</td>"
        f'<td class="status-{escape(str(call.get("status") or "unknown"))}">{escape(str(call.get("status") or "-"))}</td>'
        f"<td>{escape(_format_ms(call.get('duration_ms')))}</td>"
        f"<td><pre>{escape(str(call.get('preview') or ''))}</pre></td>"
        "</tr>"
        for call in tool_calls
    )
    if not rows:
        rows = '<tr><td colspan="5" class="muted">No tool calls.</td></tr>'
    return (
        "<h2>Tool Calls</h2><table><thead><tr>"
        "<th>Seq</th><th>Tool</th><th>Status</th><th>Duration</th><th>Preview</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _render_turn_table(turns: list[dict[str, Any]]) -> str:
    """渲染每个 Turn 的事件数、工具数、终态和短摘要。"""
    rows = "\n".join(
        "<tr>"
        f"<td><code>{escape(str(turn.get('turn_id') or '-'))}</code></td>"
        f"<td>{escape(str(turn.get('event_count') or 0))}</td>"
        f"<td>{escape(str(turn.get('tool_call_count') or 0))}</td>"
        f"<td>{escape(str(turn.get('status') or '-'))}</td>"
        f"<td><pre>{escape(str(turn.get('summary') or ''))}</pre></td>"
        "</tr>"
        for turn in turns
    )
    if not rows:
        rows = '<tr><td colspan="5" class="muted">No turns.</td></tr>'
    return (
        "<h2>Turns</h2><table><thead><tr>"
        "<th>Turn</th><th>Events</th><th>Tools</th><th>Status</th><th>Summary</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _render_event_table(events: list[dict[str, Any]]) -> str:
    """按日志顺序渲染所有事件的 seq、类型、Turn、时间与摘要。"""
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(event['seq']))}</td>"
        f'<td><span class="pill">{escape(str(event["type"]))}</span></td>'
        f"<td><code>{escape(str(event.get('turn_id') or '-'))}</code></td>"
        f"<td>{escape(str(event.get('timestamp') or '-'))}</td>"
        f"<td><pre>{escape(str(event.get('summary') or ''))}</pre></td>"
        "</tr>"
        for event in events
    )
    return (
        "<h2>Events</h2><table><thead><tr>"
        "<th>Seq</th><th>Type</th><th>Turn</th><th>Time</th><th>Summary</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _metrics(events: list[RuntimeEvent]) -> dict[str, Any]:
    """从事件事实聚合工具成功/失败、审批、错误与最终状态。"""
    tool_finishes = [
        event for event in events if event.type == RuntimeEventType.TOOL_CALL_FINISHED
    ]
    tool_failures = [
        event
        for event in tool_finishes
        if isinstance(event.payload.get("result"), dict)
        and event.payload["result"].get("success") is False
    ]
    turns = {event.turn_id for event in events if event.turn_id}
    errors = [
        event
        for event in events
        if event.type in {RuntimeEventType.ERROR, RuntimeEventType.TURN_ABORTED}
    ]
    approvals = [
        event
        for event in events
        if event.type
        in {RuntimeEventType.APPROVAL_REQUESTED, RuntimeEventType.APPROVAL_DECIDED}
    ]
    return {
        "event_count": len(events),
        "turn_count": len(turns),
        "tool_call_count": len(tool_finishes),
        "tool_success_count": len(tool_finishes) - len(tool_failures),
        "tool_failure_count": len(tool_failures),
        "approval_count": len(approvals),
        "error_count": len(errors),
        "final_status": _final_status(events),
    }


def _turns(events: list[RuntimeEvent]) -> list[dict[str, Any]]:
    """按首次出现的 turn_id 分组并汇总每轮时间、数量、状态与摘要。"""
    grouped: dict[str, list[RuntimeEvent]] = {}
    for event in events:
        if event.turn_id:
            grouped.setdefault(event.turn_id, []).append(event)

    turns = []
    for turn_id, turn_events in grouped.items():
        turns.append(
            {
                "turn_id": turn_id,
                "started_at": _iso(turn_events[0].timestamp),
                "finished_at": _iso(turn_events[-1].timestamp),
                "duration_ms": _duration_ms(
                    turn_events[0].timestamp, turn_events[-1].timestamp
                ),
                "event_count": len(turn_events),
                "tool_call_count": sum(
                    1
                    for event in turn_events
                    if event.type == RuntimeEventType.TOOL_CALL_FINISHED
                ),
                "status": _turn_status(turn_events),
                "summary": _turn_summary(turn_events),
            }
        )
    return turns


def _tool_calls(events: list[RuntimeEvent]) -> list[dict[str, Any]]:
    """用 call_id 对账模型请求、Runner 开始与完成事件，生成调用 Trace。

    名称优先取 finish，缺失时回退 started/requested；只有 FINISHED 形成一条
    调用记录，避免把尚未收口的请求误算成失败。
    """
    requested: dict[str, RuntimeEvent] = {}
    started: dict[str, RuntimeEvent] = {}
    calls: list[dict[str, Any]] = []

    for event in events:
        call_id = event.payload.get("call_id")
        if not isinstance(call_id, str):
            continue
        if event.type == RuntimeEventType.MODEL_TOOL_CALL:
            requested[call_id] = event
        elif event.type == RuntimeEventType.TOOL_CALL_STARTED:
            started[call_id] = event
        elif event.type == RuntimeEventType.TOOL_CALL_FINISHED:
            result = event.payload.get("result")
            success = isinstance(result, dict) and result.get("success") is True
            calls.append(
                {
                    "call_id": call_id,
                    "seq": event.seq,
                    "turn_id": event.turn_id,
                    "name": event.payload.get("name")
                    or _payload_name(started.get(call_id))
                    or _payload_name(requested.get(call_id)),
                    "status": "ok" if success else "failed",
                    "duration_ms": event.payload.get("duration_ms"),
                    "arguments": _payload_args(requested.get(call_id)),
                    "preview": _result_preview(result),
                    "error": result.get("error") if isinstance(result, dict) else None,
                }
            )

    return calls


def _event_row(event: RuntimeEvent) -> dict[str, Any]:
    """把类型化 RuntimeEvent 转成 Trace schema 的完整事件行。"""
    return {
        "event_id": event.event_id,
        "seq": event.seq,
        "timestamp": _iso(event.timestamp),
        "type": event.type.value,
        "turn_id": event.turn_id,
        "summary": _event_summary(event),
        "payload": event.payload.model_dump(mode="json", exclude_unset=True),
    }


def _session_config(events: list[RuntimeEvent]) -> dict[str, Any]:
    """从首事件读取 config mapping；日志为空或形态不符返回空。"""
    if not events:
        return {}
    config = events[0].payload.get("config")
    return config if isinstance(config, dict) else {}


def _final_status(events: list[RuntimeEvent]) -> str:
    """反向选择最近 Turn abort/error/finish，推导 Session Trace 终态。"""
    for event in reversed(events):
        if event.type == RuntimeEventType.TURN_ABORTED:
            return "aborted"
        if event.type == RuntimeEventType.ERROR:
            return "error"
        if event.type == RuntimeEventType.TURN_FINISHED:
            status = event.payload.get("status")
            return str(status or "success")
    return "unknown"


def _turn_status(events: list[RuntimeEvent]) -> str:
    """反向选择单 Turn 最近终态；尚无终态返回 running。"""
    for event in reversed(events):
        if event.type == RuntimeEventType.TURN_ABORTED:
            return "aborted"
        if event.type == RuntimeEventType.ERROR:
            return "error"
        if event.type == RuntimeEventType.TURN_FINISHED:
            return str(event.payload.get("status") or "success")
    return "running"


def _turn_summary(events: list[RuntimeEvent]) -> str:
    """优先用用户问题，其次最终答案，最后末事件摘要描述 Turn。"""
    for event in events:
        if event.type == RuntimeEventType.USER_MESSAGE:
            text = event.payload.get("text")
            if isinstance(text, str) and text:
                return _compact(text, 240)
    for event in reversed(events):
        if event.type == RuntimeEventType.TURN_FINISHED:
            answer = event.payload.get("answer")
            if isinstance(answer, str) and answer:
                return _compact(answer, 240)
    return _event_summary(events[-1]) if events else ""


def _event_summary(event: RuntimeEvent) -> str:
    """按事件类型提取最有信息量且长度受限的人类摘要。"""
    payload = event.payload
    if event.type == RuntimeEventType.USER_MESSAGE:
        return _compact(str(payload.get("text") or ""))
    if event.type in {
        RuntimeEventType.ASSISTANT_MESSAGE,
        RuntimeEventType.ASSISTANT_MESSAGE_DELTA,
    }:
        return _compact(str(payload.get("text") or ""))
    if event.type == RuntimeEventType.MODEL_TOOL_CALL:
        return f"{payload.get('name')} args={payload.get('arguments')}"
    if event.type == RuntimeEventType.TOOL_CALL_FINISHED:
        return _result_preview(payload.get("result"))
    if event.type in {RuntimeEventType.ERROR, RuntimeEventType.TURN_ABORTED}:
        return _compact(str(payload.get("message") or payload))
    return _compact(str(payload))


def _result_preview(result: Any) -> str:
    """从工具 result mapping 选 content/error 并压缩为短预览。"""
    if not isinstance(result, dict):
        return ""
    return _compact(str(result.get("content") or result.get("error") or ""))


def _payload_name(event: RuntimeEvent | None) -> object:
    """空安全读取事件 payload.name。"""
    return event.payload.get("name") if event else None


def _payload_args(event: RuntimeEvent | None) -> object:
    """空安全读取事件 payload.arguments。"""
    return event.payload.get("arguments") if event else None


def _compact(value: str, max_chars: int = 240) -> str:
    """合并所有空白并用省略号截成最多 max_chars 字符。"""
    compact = " ".join(value.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _duration_ms(start: datetime | None, end: datetime | None) -> int | None:
    """计算非负毫秒时差；任一端缺失返回 None。"""
    if start is None or end is None:
        return None
    return max(0, int((end - start).total_seconds() * 1000))


def _format_ms(value: object) -> str:
    """整数添加 ms，其他缺失/未知值显示短横线。"""
    return f"{value}ms" if isinstance(value, int) else "-"


def _iso(value: datetime | None) -> str | None:
    """空安全转换 datetime 为 ISO 8601。"""
    return value.isoformat() if value else None


def _join_model(session: dict[str, Any]) -> str:
    """组合 provider/model，允许任一字段缺失。"""
    provider = session.get("model_provider")
    model = session.get("model")
    if provider and model:
        return f"{provider}/{model}"
    return str(model or provider or "-")
