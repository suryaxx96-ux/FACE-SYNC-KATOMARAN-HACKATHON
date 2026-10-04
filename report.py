"""Unique visitor count from the DB, cross-checked against the count derived from events.log.

Usage: python report.py [db_path] [log_path]
"""
import os
import json
import re
import sqlite3
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent
db_path = sys.argv[1] if len(sys.argv) > 1 else str(project_dir / "data" / "visitors.db")
log_path = sys.argv[2] if len(sys.argv) > 2 else str(project_dir / "logs" / "events.log")
conn = sqlite3.connect(db_path)
print("Database file:", os.path.abspath(db_path))

db_count = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
print("Unique visitors (database):", db_count)

if os.path.exists(log_path):
    with open(log_path, encoding="utf-8") as f:
        text = f.read()
    # Support both the original registration line and the structured audit line.
    registered = set(re.findall(r"REGISTER \| new face=(FACE_\d+)", text))
    registered.update(re.findall(
        r"CRITICAL EVENT(?: RECOVERED)? \| audit_id=\d+ \| type=REGISTRATION \| face=(FACE_\d+)",
        text,
    ))
    log_count = len(registered)
    print("Unique visitors (derived from events.log):", log_count)
    print("Database and log agree." if log_count == db_count else "WARNING: database and log counts differ!")

print("\nPer-face entry/exit timeline:")
events_by_face = {}
for face_id, event_type, timestamp in conn.execute(
    "SELECT face_id, event_type, timestamp FROM events ORDER BY face_id, timestamp, event_id"
):
    events_by_face.setdefault(face_id, {"entry": [], "exit": []})[event_type].append(timestamp)

for face_id, label, first_seen in conn.execute(
    "SELECT face_id, label, first_seen FROM faces ORDER BY face_id"
):
    face_events = events_by_face.get(face_id, {"entry": [], "exit": []})
    entries, exits = face_events["entry"], face_events["exit"]
    print(f"  {label} (face_id={face_id}) | first_seen={first_seen}")
    print(f"    Entry times ({len(entries)}):")
    for timestamp in entries:
        print(f"      - {timestamp}")
    if not entries:
        print("      - none recorded")
    print(f"    Exit times ({len(exits)}):")
    for timestamp in exits:
        print(f"      - {timestamp}")
    if not exits:
        print("      - none recorded")

print("\nEvents:")
for row in conn.execute(
    "SELECT timestamp, event_type, face_id, track_id, frame_idx, image_path FROM events ORDER BY event_id"
):
    print("  ", *row, sep=" | ")

has_audit_table = conn.execute(
    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='system_events'"
).fetchone()
if has_audit_table:
    print("\nCritical system events (database audit):")
    for event_type, count in conn.execute(
        "SELECT event_type, COUNT(*) FROM system_events GROUP BY event_type ORDER BY event_type"
    ):
        print(f"  {event_type}: {count}")
    print("\nAudit rows:")
    audit_sql = (
        "SELECT audit_id, event_type, timestamp, face_id, track_id, frame_idx, image_path, metadata "
        "FROM system_events ORDER BY audit_id"
    )
    for row in conn.execute(audit_sql):
        audit_id, event_type, timestamp, face_id, track_id, frame_idx, image_path, metadata = row
        face = f"FACE_{face_id:04d}" if face_id is not None else "-"
        try:
            metadata = json.dumps(json.loads(metadata), sort_keys=True)
        except (TypeError, json.JSONDecodeError):
            pass
        print(
            f"  audit_id={audit_id} | type={event_type} | timestamp={timestamp} "
            f"| face={face} | track={track_id} | frame={frame_idx} "
            f"| image={image_path or '-'} | metadata={metadata}"
        )
else:
    print("\nCritical system-event audit is not present in this older database yet.")
conn.close()
