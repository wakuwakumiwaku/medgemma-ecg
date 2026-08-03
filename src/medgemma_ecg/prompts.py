from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "You are analyzing a de-identified 12-lead ECG for a research dataset. "
    "Return only the requested JSON. Do not invent measurements that are not visible."
)

USER_PROMPT = (
    "Analyze this 12-lead ECG systematically. Return one JSON object with exactly "
    "these keys: rhythm, rate_bpm, axis, intervals, findings, labels, impression, "
    "uncertainty. Use null when a value cannot be determined. Use arrays for findings "
    "and labels. Do not add prose outside JSON."
)

TARGET_KEYS = (
    "rhythm",
    "rate_bpm",
    "axis",
    "intervals",
    "findings",
    "labels",
    "impression",
    "uncertainty",
)


def normalize_target(target: dict) -> dict:
    """Return a target with stable keys and JSON-compatible values."""
    normalized = {key: target.get(key) for key in TARGET_KEYS}
    normalized["findings"] = list(normalized["findings"] or [])
    normalized["labels"] = sorted(set(normalized["labels"] or []))
    return normalized


def target_text(target: dict) -> str:
    return json.dumps(normalize_target(target), ensure_ascii=False, separators=(",", ":"))


def training_messages(target: dict | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]
    if target is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text(target)}],
            }
        )
    return messages
