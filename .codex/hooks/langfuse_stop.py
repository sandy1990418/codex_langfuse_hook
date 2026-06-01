#!/usr/bin/env python3
"""
Codex Stop hook -> Langfuse.

On each completed Codex turn, read the hook payload/transcript and emit
one Langfuse trace with a generation observation and tool events.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MAX_CHARS = 0  # 0 means no truncation.
DEFAULT_MAX_TOOL_EVENTS = 0  # 0 means no tool-event limit.
STATE_FILE = Path.home() / ".codex" / "state" / "langfuse_codex_hook_state.json"
LOG_FILE = Path.home() / ".codex" / "state" / "langfuse_codex_hook.log"

SECRET_KEY_RE = re.compile(
    r"(secret|password|token|api[_-]?key|access[_-]?key|authorization)",
    re.IGNORECASE,
)
NON_SECRET_KEYS = {
    "max_output_tokens",
    "model_context_window",
    "last_token_usage",
    "total_token_usage",
    "token_count",
    "token_usage",
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "completion_tokens",
    "redact_secrets",
}
SECRET_VALUE_PATTERNS = [
    re.compile(r"sk-lf-[A-Za-z0-9-]+"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)(LANGFUSE_SECRET_KEY\s*=\s*)([\"']?)[^\"'\s]+"),
    re.compile(r"(?i)(OPENAI_API_KEY\s*=\s*)([\"']?)[^\"'\s]+"),
]


def _json_response() -> None:
    print(json.dumps({"continue": True}, separators=(",", ":")))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _max_chars() -> int:
    return _env_int("CODEX_LANGFUSE_MAX_CHARS", DEFAULT_MAX_CHARS)


def _max_tool_events() -> int:
    return _env_int("CODEX_LANGFUSE_MAX_TOOL_EVENTS", DEFAULT_MAX_TOOL_EVENTS)


def _redaction_enabled() -> bool:
    return _env_bool("CODEX_LANGFUSE_REDACT_SECRETS", True)


def _capture_timeline_enabled() -> bool:
    return _env_bool("CODEX_LANGFUSE_CAPTURE_TIMELINE", True)


def _capture_raw_turn_enabled() -> bool:
    return _env_bool("CODEX_LANGFUSE_CAPTURE_RAW_TURN", False)


def _log(message: str) -> None:
    if os.environ.get("CODEX_LANGFUSE_DEBUG", "").lower() != "true":
        return
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")
    except Exception:
        pass


def _read_stdin() -> dict[str, Any]:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return {}
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        _log(f"failed to parse stdin: {exc!r}")
        return {}


def _load_dotenv(cwd: Path) -> None:
    for env_path in (
        cwd / ".env.codex-langfuse",
        Path.home() / ".codex" / "langfuse.env",
    ):
        if not env_path.exists():
            continue
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                key = key.strip()
                if not key or key in os.environ:
                    continue
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
        except Exception as exc:
            _log(f"failed to load {env_path}: {exc!r}")


def _redact_string(value: str) -> str:
    if not _redaction_enabled():
        return value

    redacted = value
    for pattern in SECRET_VALUE_PATTERNS:
        redacted = pattern.sub(
            lambda m: (
                f"{m.group(1)}***REDACTED***" if m.groups() else "***REDACTED***"
            ),
            redacted,
        )
    return redacted


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    if normalized in NON_SECRET_KEYS or "public" in normalized:
        return False
    if normalized.endswith("_tokens") or normalized.endswith("_token_usage"):
        return False
    return bool(SECRET_KEY_RE.search(normalized))


def _truncate(value: str, max_chars: int | None = None) -> str:
    if max_chars is None:
        max_chars = _max_chars()
    if max_chars <= 0:
        return value
    if len(value) <= max_chars:
        return value
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
    return (
        value[:max_chars]
        + f"\n[truncated by codex-langfuse-hook: original_chars={len(value)} sha256={digest}]"
    )


def _sanitize(value: Any, max_chars: int | None = None) -> Any:
    if max_chars is None:
        max_chars = _max_chars()
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if (
                _redaction_enabled()
                and _is_secret_key(key_str)
            ):
                sanitized[key_str] = "***REDACTED***"
            else:
                sanitized[key_str] = _sanitize(item, max_chars=max_chars)
        return sanitized
    if isinstance(value, list):
        return [_sanitize(item, max_chars=max_chars) for item in value]
    if isinstance(value, str):
        return _truncate(_redact_string(value), max_chars=max_chars)
    return value


def _load_state() -> dict[str, Any]:
    try:
        if STATE_FILE.exists():
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _save_state(state: dict[str, Any]) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, STATE_FILE)
    except Exception as exc:
        _log(f"failed to save state: {exc!r}")


def _parse_json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return ""


def _read_transcript(path_value: Any) -> list[dict[str, Any]]:
    if not isinstance(path_value, str) or not path_value:
        return []
    path = Path(path_value).expanduser()
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    records.append(row)
    except Exception as exc:
        _log(f"failed to read transcript {path}: {exc!r}")
    return records


def _payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("payload")
    return payload if isinstance(payload, dict) else {}


def _turn_records(
    records: list[dict[str, Any]], turn_id: str | None
) -> list[dict[str, Any]]:
    if not records:
        return []

    start = 0
    if turn_id:
        for idx, record in enumerate(records):
            payload = _payload(record)
            if payload.get("type") == "task_started" and payload.get("turn_id") == turn_id:
                start = idx
    else:
        for idx, record in enumerate(records):
            if _payload(record).get("type") == "task_started":
                start = idx

    end = len(records)
    for idx in range(start + 1, len(records)):
        if _payload(records[idx]).get("type") == "task_started":
            end = idx
            break
    return records[start:end]


def _extract_prompt(records: list[dict[str, Any]]) -> str:
    prompt = ""
    for record in records:
        payload = _payload(record)
        if payload.get("type") == "user_message":
            prompt = str(payload.get("message") or "")
        elif (
            record.get("type") == "response_item"
            and payload.get("type") == "message"
            and payload.get("role") == "user"
        ):
            prompt = _text_from_content(payload.get("content")) or prompt
    return prompt


def _extract_assistant(records: list[dict[str, Any]], fallback: Any) -> str:
    if isinstance(fallback, str) and fallback.strip():
        return fallback

    assistant = ""
    for record in records:
        payload = _payload(record)
        if payload.get("type") == "agent_message":
            assistant = str(payload.get("message") or assistant)
        elif (
            record.get("type") == "response_item"
            and payload.get("type") == "message"
            and payload.get("role") == "assistant"
        ):
            assistant = _text_from_content(payload.get("content")) or assistant
    return assistant


def _extract_codex_prompts(
    transcript_records: list[dict[str, Any]],
    current_records: list[dict[str, Any]],
) -> dict[str, Any]:
    base_instructions: Any = None
    session_metadata: dict[str, Any] = {}
    for record in transcript_records:
        if record.get("type") != "session_meta":
            continue
        payload = _payload(record)
        base_instructions = payload.get("base_instructions")
        session_metadata = {
            "session_id": payload.get("id"),
            "cli_version": payload.get("cli_version"),
            "originator": payload.get("originator"),
            "model_provider": payload.get("model_provider"),
            "source": payload.get("source"),
            "thread_source": payload.get("thread_source"),
        }
        break

    turn_context: dict[str, Any] = {}
    for record in current_records:
        if record.get("type") == "turn_context":
            turn_context = _payload(record)
            break

    collaboration_mode = turn_context.get("collaboration_mode")
    collaboration_settings = (
        collaboration_mode.get("settings")
        if isinstance(collaboration_mode, dict)
        else {}
    )
    developer_instructions = (
        collaboration_settings.get("developer_instructions")
        if isinstance(collaboration_settings, dict)
        else None
    )

    system_messages: list[dict[str, Any]] = []
    developer_messages: list[dict[str, Any]] = []
    context_messages: list[dict[str, Any]] = []
    for record in current_records:
        payload = _payload(record)
        if record.get("type") != "response_item" or payload.get("type") != "message":
            continue

        role = payload.get("role")
        message = {
            "timestamp": record.get("timestamp"),
            "role": role,
            "content": payload.get("content"),
        }
        if role == "system":
            system_messages.append(message)
        elif role == "developer":
            developer_messages.append(message)
        elif role == "user":
            text = _text_from_content(payload.get("content"))
            if text.startswith("<environment_context>"):
                context_messages.append(message)

    return {
        "base_instructions": base_instructions,
        "developer_instructions": developer_instructions,
        "system_messages": system_messages,
        "developer_messages": developer_messages,
        "context_messages": context_messages,
        "turn_context": turn_context,
        "session_metadata": session_metadata,
    }


def _extract_tool_calls(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    outputs: dict[str, dict[str, Any]] = {}

    for record in records:
        payload = _payload(record)
        payload_type = payload.get("type")
        if record.get("type") != "response_item":
            continue
        if payload_type in {"function_call_output", "custom_tool_call_output"}:
            call_id = str(payload.get("call_id") or "")
            if call_id:
                outputs[call_id] = {
                    "output": payload.get("output"),
                    "timestamp": record.get("timestamp"),
                }

    for record in records:
        payload = _payload(record)
        payload_type = payload.get("type")
        if record.get("type") != "response_item" or payload_type not in {
            "function_call",
            "custom_tool_call",
        }:
            continue
        call_id = str(payload.get("call_id") or "")
        output_record = outputs.get(call_id) or {}
        output = output_record.get("output")
        status = _tool_status(payload.get("name") or "tool", output)
        calls.append(
            {
                "call_id": call_id,
                "name": payload.get("name") or "tool",
                "input": _parse_json_maybe(payload.get("arguments") or payload.get("input")),
                "output": output,
                "timestamp": record.get("timestamp"),
                "end_timestamp": output_record.get("timestamp") or record.get("timestamp"),
                "source_record_type": payload_type,
                "status": status,
            }
        )

    for record in records:
        payload = _payload(record)
        if record.get("type") != "event_msg" or payload.get("type") != "web_search_end":
            continue
        calls.append(
            {
                "call_id": payload.get("call_id"),
                "name": "web_search",
                "input": {
                    "query": payload.get("query"),
                    "action": payload.get("action"),
                },
                "output": {
                    "status": "completed",
                    "query": payload.get("query"),
                },
                "timestamp": record.get("timestamp"),
                "end_timestamp": record.get("timestamp"),
                "source_record_type": "web_search_end",
                "status": {
                    "status": "completed",
                    "level": "DEFAULT",
                    "status_message": "completed",
                },
            }
        )
    return calls


def _tool_status(name: str, output: Any) -> dict[str, Any]:
    status: dict[str, Any] = {
        "status": "completed",
        "level": "DEFAULT",
        "status_message": "completed",
    }

    if isinstance(output, str):
        exit_match = re.search(r"Process exited with code (-?\d+)", output)
        if exit_match:
            exit_code = int(exit_match.group(1))
            status["exit_code"] = exit_code
            if exit_code == 0:
                status["status_message"] = "exit_code=0"
            else:
                status["status"] = "failed"
                status["level"] = "ERROR"
                status["status_message"] = f"exit_code={exit_code}"

        wall_match = re.search(r"Wall time:\s*([0-9.]+)\s*seconds", output)
        if wall_match:
            status["duration_seconds"] = float(wall_match.group(1))

    if name == "exec_command" and "exit_code" not in status:
        status["status"] = "unknown"
        status["level"] = "WARNING"
        status["status_message"] = "missing exit code"

    return status


def _latest_token_info(records: list[dict[str, Any]]) -> Any:
    latest = None
    for record in records:
        payload = _payload(record)
        if payload.get("type") == "token_count":
            latest = payload.get("info")
    return latest


def _build_timeline(
    records: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tools_by_call_id = {
        str(tool.get("call_id")): tool for tool in tool_calls if tool.get("call_id")
    }
    timeline: list[dict[str, Any]] = []
    latest_token_count: dict[str, Any] | None = None
    emitted_tool_call_ids: set[str] = set()
    emitted_turn_context = False
    reasoning_summary: dict[str, Any] = {
        "type": "reasoning_summary",
        "count": 0,
        "encrypted_content_count": 0,
        "encrypted_content_chars": 0,
        "summaries": [],
    }

    for record in records:
        payload = _payload(record)
        payload_type = payload.get("type")
        timestamp = record.get("timestamp")

        if payload_type == "task_started":
            timeline.append(
                {
                    "type": "task_started",
                    "timestamp": timestamp,
                    "turn_id": payload.get("turn_id"),
                    "started_at": payload.get("started_at"),
                    "model_context_window": payload.get("model_context_window"),
                    "collaboration_mode_kind": payload.get(
                        "collaboration_mode_kind"
                    ),
                }
            )
        elif record.get("type") == "turn_context" and not emitted_turn_context:
            emitted_turn_context = True
            timeline.append(
                {
                    "type": "turn_context",
                    "timestamp": timestamp,
                    "cwd": payload.get("cwd"),
                    "model": payload.get("model"),
                    "current_date": payload.get("current_date"),
                    "timezone": payload.get("timezone"),
                    "approval_policy": payload.get("approval_policy"),
                    "sandbox_policy": payload.get("sandbox_policy"),
                    "effort": payload.get("effort"),
                    "summary": payload.get("summary"),
                }
            )
        elif payload_type == "user_message":
            timeline.append(
                {
                    "type": "user_message",
                    "timestamp": timestamp,
                    "text": payload.get("message"),
                    "images": len(payload.get("images") or []),
                    "local_images": len(payload.get("local_images") or []),
                    "text_elements": len(payload.get("text_elements") or []),
                }
            )
        elif payload_type == "agent_message":
            timeline.append(
                {
                    "type": "assistant_message",
                    "timestamp": timestamp,
                    "phase": payload.get("phase"),
                    "text": payload.get("message"),
                }
            )
        elif payload_type in {"function_call", "custom_tool_call"}:
            call_id = str(payload.get("call_id") or "")
            tool = tools_by_call_id.get(call_id)
            emitted_tool_call_ids.add(call_id)
            timeline.append(
                {
                    "type": "tool_call",
                    "timestamp": timestamp,
                    "call_id": call_id,
                    "name": payload.get("name") or "tool",
                    "input": _parse_json_maybe(
                        payload.get("arguments") or payload.get("input")
                    ),
                    "output": tool.get("output") if tool else None,
                    "status": (tool.get("status") if tool else None),
                }
            )
        elif payload_type == "web_search_end":
            call_id = str(payload.get("call_id") or "")
            tool = tools_by_call_id.get(call_id)
            emitted_tool_call_ids.add(call_id)
            timeline.append(
                {
                    "type": "tool_call",
                    "timestamp": timestamp,
                    "call_id": call_id,
                    "name": "web_search",
                    "input": {
                        "query": payload.get("query"),
                        "action": payload.get("action"),
                    },
                    "output": tool.get("output") if tool else None,
                    "status": (tool.get("status") if tool else None),
                }
            )
        elif payload_type == "reasoning":
            encrypted_content = payload.get("encrypted_content")
            reasoning_summary["count"] += 1
            reasoning_summary["first_timestamp"] = (
                reasoning_summary.get("first_timestamp") or timestamp
            )
            reasoning_summary["last_timestamp"] = timestamp
            if encrypted_content:
                reasoning_summary["encrypted_content_count"] += 1
                reasoning_summary["encrypted_content_chars"] += len(encrypted_content)
            summary = payload.get("summary")
            content = payload.get("content")
            if summary or content:
                reasoning_summary["summaries"].append(
                    {
                        "timestamp": timestamp,
                        "summary": summary,
                        "content": content,
                    }
                )
        elif payload_type == "token_count":
            latest_token_count = {
                "type": "token_count_final",
                "timestamp": timestamp,
                "info": payload.get("info"),
                "rate_limits": payload.get("rate_limits"),
            }
        elif payload_type == "task_complete":
            timeline.append(
                {
                    "type": "task_complete",
                    "timestamp": timestamp,
                    "duration_ms": payload.get("duration_ms"),
                }
            )

    for tool in tool_calls:
        call_id = str(tool.get("call_id") or "")
        if call_id in emitted_tool_call_ids:
            continue
        timeline.append(
            {
                "type": "tool_call",
                "timestamp": tool.get("timestamp"),
                "call_id": call_id,
                "name": tool.get("name") or "tool",
                "input": tool.get("input"),
                "output": tool.get("output"),
                "status": tool.get("status"),
            }
        )

    if reasoning_summary["count"]:
        timeline.append(reasoning_summary)
    if latest_token_count:
        timeline.append(latest_token_count)
    return timeline


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event_timestamp(value: Any) -> str:
    return value if isinstance(value, str) and value else _now()


def _send_to_langfuse(payload: dict[str, Any]) -> bool:
    if os.environ.get("CODEX_LANGFUSE_DISABLED", "").lower() == "true":
        _log("disabled by CODEX_LANGFUSE_DISABLED")
        return False

    cwd = Path(payload.get("cwd") or os.getcwd())
    _load_dotenv(cwd)

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    base_url = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_BASEURL")
    if not public_key or not secret_key:
        _log("missing Langfuse keys")
        return False

    session_id = str(payload.get("session_id") or "")
    turn_id = str(payload.get("turn_id") or "")
    if not session_id and not turn_id:
        _log("missing session_id and turn_id")
        return False

    dedupe_key = f"{session_id}:{turn_id}"
    state = _load_state()
    if state.get(dedupe_key):
        _log(f"already sent {dedupe_key}")
        return True

    transcript_path = payload.get("transcript_path")
    transcript_records = _read_transcript(transcript_path)
    current_records = _turn_records(transcript_records, turn_id)
    prompt = _extract_prompt(current_records)
    assistant = _extract_assistant(current_records, payload.get("last_assistant_message"))
    codex_prompts = _extract_codex_prompts(transcript_records, current_records)
    tool_calls = _extract_tool_calls(current_records)
    token_info = _latest_token_info(current_records)
    max_chars = _max_chars()
    max_tool_events = _max_tool_events()
    selected_tool_calls = (
        tool_calls if max_tool_events <= 0 else tool_calls[:max_tool_events]
    )
    timeline = _build_timeline(current_records, selected_tool_calls)

    trace_id = str(uuid.uuid4())
    generation_id = str(uuid.uuid4())
    timestamp = _now()
    metadata = _sanitize(
        {
            "source": "codex-hook",
            "hook_event_name": payload.get("hook_event_name"),
            "turn_id": turn_id,
            "cwd": payload.get("cwd"),
            "permission_mode": payload.get("permission_mode"),
            "transcript_path": transcript_path,
            "tool_call_count": len(tool_calls),
            "captured_tool_event_count": len(selected_tool_calls),
            "raw_turn_record_count": len(current_records),
            "timeline_event_count": len(timeline),
            "capture_timeline": _capture_timeline_enabled(),
            "capture_raw_turn": _capture_raw_turn_enabled(),
            "max_chars": max_chars,
            "max_tool_events": max_tool_events,
            "redact_secrets": _redaction_enabled(),
            "token_count": token_info,
            "base_url": base_url,
        }
    )
    generation_input = {
        "system": codex_prompts.get("base_instructions"),
        "developer": {
            "developer_instructions": codex_prompts.get("developer_instructions"),
            "system_messages": codex_prompts.get("system_messages"),
            "developer_messages": codex_prompts.get("developer_messages"),
            "context_messages": codex_prompts.get("context_messages"),
        },
        "user": prompt,
    }
    trace_input = _sanitize(generation_input, max_chars=max_chars)
    trace_output = _sanitize(assistant, max_chars=max_chars)
    start_time = _event_timestamp(
        current_records[0].get("timestamp") if current_records else None
    )

    batch: list[dict[str, Any]] = [
        {
            "type": "trace-create",
            "id": str(uuid.uuid4()),
            "timestamp": timestamp,
            "body": {
                "id": trace_id,
                "name": "codex-turn",
                "sessionId": session_id or None,
                "userId": os.environ.get("USER"),
                "input": trace_input,
                "output": trace_output,
                "metadata": metadata,
                "timestamp": timestamp,
                "tags": ["codex", "codex-hook"],
            },
        },
        {
            "type": "generation-create",
            "id": str(uuid.uuid4()),
            "timestamp": timestamp,
            "body": {
                "id": generation_id,
                "traceId": trace_id,
                "name": "codex-assistant",
                "startTime": start_time,
                "endTime": timestamp,
                "model": payload.get("model"),
                "input": trace_input,
                "output": trace_output,
                "metadata": metadata,
            },
        },
    ]

    batch.append(
        {
            "type": "event-create",
            "id": str(uuid.uuid4()),
            "timestamp": timestamp,
            "body": {
                "id": str(uuid.uuid4()),
                "traceId": trace_id,
                "name": "codex:prompts",
                "startTime": start_time,
                "input": _sanitize(codex_prompts, max_chars=max_chars),
                "metadata": _sanitize(
                    {
                        "turn_id": turn_id,
                        "source": "codex-hook",
                        "note": "Codex base/system/developer/context prompt material.",
                    }
                ),
            },
        }
    )

    if _capture_timeline_enabled():
        batch.append(
            {
                "type": "event-create",
                "id": str(uuid.uuid4()),
                "timestamp": timestamp,
                "body": {
                    "id": str(uuid.uuid4()),
                    "traceId": trace_id,
                    "name": "codex:timeline",
                    "startTime": start_time,
                    "input": _sanitize(timeline, max_chars=max_chars),
                    "metadata": _sanitize(
                        {
                            "turn_id": turn_id,
                            "source": "codex-hook",
                            "record_count": len(timeline),
                            "note": (
                                "Readable Codex turn timeline with duplicated "
                                "transcript layers collapsed."
                            ),
                        }
                    ),
                },
            }
        )

    if _capture_raw_turn_enabled():
        batch.append(
            {
                "type": "event-create",
                "id": str(uuid.uuid4()),
                "timestamp": timestamp,
                "body": {
                    "id": str(uuid.uuid4()),
                    "traceId": trace_id,
                    "name": "codex:raw-turn-records",
                    "startTime": start_time,
                    "input": _sanitize(current_records, max_chars=max_chars),
                    "metadata": _sanitize(
                        {
                            "turn_id": turn_id,
                            "source": "codex-hook",
                            "record_count": len(current_records),
                            "note": "Full Codex JSONL records for this turn.",
                        }
                    ),
                },
            }
        )

    for tool in selected_tool_calls:
        tool_status = tool.get("status") or {}
        batch.append(
            {
                "type": "observation-create",
                "id": str(uuid.uuid4()),
                "timestamp": _event_timestamp(tool.get("timestamp")),
                "body": {
                    "id": str(uuid.uuid4()),
                    "traceId": trace_id,
                    "type": "TOOL",
                    "name": f"tool:{tool.get('name') or 'tool'}",
                    "startTime": _event_timestamp(tool.get("timestamp")),
                    "endTime": _event_timestamp(
                        tool.get("end_timestamp") or tool.get("timestamp")
                    ),
                    "parentObservationId": generation_id,
                    "input": _sanitize(tool.get("input"), max_chars=max_chars),
                    "output": _sanitize(tool.get("output"), max_chars=max_chars),
                    "level": tool_status.get("level", "DEFAULT"),
                    "statusMessage": tool_status.get("status_message"),
                    "metadata": _sanitize(
                        {
                            "call_id": tool.get("call_id"),
                            "turn_id": turn_id,
                            "source": "codex-hook",
                            "source_record_type": tool.get("source_record_type"),
                            "status": tool_status.get("status"),
                            "exit_code": tool_status.get("exit_code"),
                            "duration_seconds": tool_status.get("duration_seconds"),
                        }
                    ),
                },
            }
        )

    if os.environ.get("CODEX_LANGFUSE_DRY_RUN", "").lower() == "true":
        _log(f"dry run prepared {len(batch)} events")
        return True

    try:
        from langfuse import Langfuse  # type: ignore[import]

        kwargs: dict[str, Any] = {
            "public_key": public_key,
            "secret_key": secret_key,
            "timeout": 5,
        }
        if base_url:
            kwargs["base_url"] = base_url
        try:
            client = Langfuse(**kwargs)
        except TypeError:
            if base_url:
                kwargs.pop("base_url", None)
                kwargs["host"] = base_url
            client = Langfuse(**kwargs)
        client.api.ingestion.batch(batch=batch)
        with suppress(Exception):
            client.flush()
        state[dedupe_key] = {"trace_id": trace_id, "sent_at": timestamp}
        _save_state(state)
        _log(f"sent trace {trace_id} with {len(batch)} events")
        return True
    except Exception as exc:
        _log(f"send failed: {exc!r}")
        return False


def main() -> None:
    payload = _read_stdin()
    _send_to_langfuse(payload)
    _json_response()


if __name__ == "__main__":
    main()
