"""Context-window-friendly logging (Phase 6, rule 4).

Heavy output goes to logs/training.log; scripts print only compact summaries
to stdout. Read the tail of the log file, never the whole thing.
"""
from datetime import datetime

from config import LOG_FILE


def log(msg: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{stamp} {msg}\n")
