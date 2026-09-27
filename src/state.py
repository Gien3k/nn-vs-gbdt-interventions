"""Persistent state for the discovery loop (Phase 6 protocol).

discovery_state.json is the single source of truth for session resumption.
Writes are atomic (temp file + os.replace) so a crash mid-write can never
corrupt the state. Every mutation appends to the history log inside the file
so the full trajectory of the investigation is auditable.
"""
import json
import os
import tempfile
from datetime import datetime, timezone

from config import STATE_FILE

_DEFAULT = {
    "project": "NN-vs-GBDT zero-cost meta-learner (target: Neural Computing and Applications, 100 pkt MEiN)",
    "phase": "3-setup",
    "current_hypothesis": None,
    "best_finding": None,
    "completed_datasets": [],
    "failed_datasets": {},
    "next_steps": [],
    "history": [],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load() -> dict:
    if not STATE_FILE.exists():
        return dict(_DEFAULT)
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save(state: dict) -> None:
    """Atomic write: never leaves a half-written state file."""
    state["updated_at"] = _now()
    fd, tmp = tempfile.mkstemp(dir=STATE_FILE.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(tmp, STATE_FILE)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def update(**kwargs) -> dict:
    """Merge kwargs into state, stamp history, persist. Returns new state."""
    state = load()
    note = kwargs.pop("note", None)
    state.update(kwargs)
    if note:
        state["history"].append({"t": _now(), "note": note})
    save(state)
    return state


def mark_dataset_done(dataset_id: int, note: str | None = None) -> None:
    state = load()
    if dataset_id not in state["completed_datasets"]:
        state["completed_datasets"].append(dataset_id)
    if note:
        state["history"].append({"t": _now(), "note": note})
    save(state)


def mark_dataset_failed(dataset_id: int, reason: str) -> None:
    state = load()
    state["failed_datasets"][str(dataset_id)] = reason
    save(state)
