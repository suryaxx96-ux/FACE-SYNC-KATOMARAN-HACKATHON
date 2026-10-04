"""SQLite persistence: registered faces (+ embeddings) and entry/exit events.

WAL mode + a commit per write keeps the DB consistent if the process is killed.
"""
import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Dict, Optional

import numpy as np


def face_label(face_id: int) -> str:
    return f"FACE_{face_id:04d}"


class Database:
    def __init__(self, path: str, reset: bool = False):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Do not unlink an open SQLite file on Windows. The dashboard/report may
        # have a reader open, so reset rows transactionally after the schema opens.
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.lock = threading.Lock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            # FULL asks SQLite to flush each committed transaction to stable storage.
            # WAL still allows readers while protecting committed events on interruption.
            self.conn.execute("PRAGMA synchronous=FULL")
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS faces (
                    face_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    label      TEXT,
                    first_seen TEXT NOT NULL,
                    embedding  BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id    INTEGER NOT NULL,
                    event_type TEXT NOT NULL CHECK (event_type IN ('entry','exit')),
                    timestamp  TEXT NOT NULL,
                    image_path TEXT,
                    track_id   INTEGER,
                    frame_idx  INTEGER,
                    FOREIGN KEY (face_id) REFERENCES faces(face_id)
                );
                CREATE TABLE IF NOT EXISTS system_events (
                    audit_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    timestamp  TEXT NOT NULL,
                    face_id    INTEGER,
                    track_id   INTEGER,
                    frame_idx  INTEGER,
                    image_path TEXT,
                    metadata   TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (face_id) REFERENCES faces(face_id)
                );
                CREATE INDEX IF NOT EXISTS idx_events_face ON events(face_id);
                CREATE INDEX IF NOT EXISTS idx_system_events_type_time
                    ON system_events(event_type, timestamp);
                """
            )
            self.conn.commit()
            if reset:
                try:
                    self.conn.execute("BEGIN IMMEDIATE")
                    self.conn.execute("DELETE FROM events")
                    self.conn.execute("DELETE FROM system_events")
                    self.conn.execute("DELETE FROM faces")
                    self.conn.execute(
                        "DELETE FROM sqlite_sequence WHERE name IN ('faces','events','system_events')"
                    )
                    self.conn.commit()
                except Exception:
                    self.conn.rollback()
                    raise

    # ---- faces ----
    def add_face(self, embedding: np.ndarray, ts: datetime) -> int:
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO faces (first_seen, embedding) VALUES (?, ?)",
                (ts.isoformat(timespec="milliseconds"), embedding.astype(np.float32).tobytes()),
            )
            fid = cur.lastrowid
            self.conn.execute("UPDATE faces SET label=? WHERE face_id=?", (face_label(fid), fid))
            self.conn.commit()
            return fid

    def register_face(self, embedding: np.ndarray, ts: datetime, track_id: int,
                      frame_idx: int, metadata: Optional[dict] = None) -> tuple:
        """Create the face and its registration audit record atomically."""
        with self.lock:
            try:
                cur = self.conn.execute(
                    "INSERT INTO faces (first_seen, embedding) VALUES (?, ?)",
                    (ts.isoformat(timespec="milliseconds"),
                     embedding.astype(np.float32).tobytes()),
                )
                face_id = int(cur.lastrowid)
                self.conn.execute(
                    "UPDATE faces SET label=? WHERE face_id=?",
                    (face_label(face_id), face_id),
                )
                details = dict(metadata or {})
                details.setdefault("summary", "New face registered")
                audit_cur = self.conn.execute(
                    "INSERT INTO system_events "
                    "(event_type, timestamp, face_id, track_id, frame_idx, metadata) "
                    "VALUES (?,?,?,?,?,?)",
                    ("registration", ts.isoformat(timespec="milliseconds"), face_id,
                     track_id, frame_idx,
                     json.dumps(details, sort_keys=True, separators=(",", ":"))),
                )
                self.conn.commit()
                return face_id, int(audit_cur.lastrowid)
            except Exception:
                self.conn.rollback()
                raise

    def update_embedding(self, face_id: int, embedding: np.ndarray) -> None:
        with self.lock:
            self.conn.execute(
                "UPDATE faces SET embedding=? WHERE face_id=?",
                (embedding.astype(np.float32).tobytes(), face_id),
            )
            self.conn.commit()

    def update_embedding_with_audit(self, face_id: int, embedding: np.ndarray,
                                    ts: datetime, track_id: int, frame_idx: int,
                                    metadata: Optional[dict] = None) -> int:
        """Update a face embedding and record that change in one transaction."""
        with self.lock:
            try:
                self.conn.execute(
                    "UPDATE faces SET embedding=? WHERE face_id=?",
                    (embedding.astype(np.float32).tobytes(), face_id),
                )
                cur = self.conn.execute(
                    "INSERT INTO system_events "
                    "(event_type, timestamp, face_id, track_id, frame_idx, metadata) "
                    "VALUES (?,?,?,?,?,?)",
                    ("embedding_gallery_update", ts.isoformat(timespec="milliseconds"),
                     face_id, track_id, frame_idx,
                     json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"))),
                )
                self.conn.commit()
                return int(cur.lastrowid)
            except Exception:
                self.conn.rollback()
                raise

    def load_faces(self) -> Dict[int, np.ndarray]:
        with self.lock:
            rows = self.conn.execute("SELECT face_id, embedding FROM faces").fetchall()
        return {fid: np.frombuffer(blob, dtype=np.float32).copy() for fid, blob in rows}

    # ---- events ----
    def add_event(self, face_id: int, event_type: str, ts: datetime,
                  image_path: Optional[str], track_id: int, frame_idx: int) -> tuple:
        with self.lock:
            timestamp = ts.isoformat(timespec="milliseconds")
            try:
                cur = self.conn.execute(
                    "INSERT INTO events (face_id, event_type, timestamp, image_path, track_id, frame_idx) "
                    "VALUES (?,?,?,?,?,?)",
                    (face_id, event_type, timestamp, image_path, track_id, frame_idx),
                )
                event_id = int(cur.lastrowid)
                audit_cur = self.conn.execute(
                    "INSERT INTO system_events "
                    "(event_type, timestamp, face_id, track_id, frame_idx, image_path, metadata) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (event_type, timestamp, face_id, track_id, frame_idx, image_path,
                     json.dumps({"event_id": event_id}, separators=(",", ":"))),
                )
                self.conn.commit()
                return event_id, int(audit_cur.lastrowid)
            except Exception:
                self.conn.rollback()
                raise

    def add_audit_event(self, event_type: str, ts: datetime,
                        face_id: Optional[int] = None,
                        track_id: Optional[int] = None,
                        frame_idx: Optional[int] = None,
                        metadata: Optional[dict] = None) -> int:
        """Persist a critical pipeline event and its structured metadata."""
        with self.lock:
            try:
                cur = self.conn.execute(
                    "INSERT INTO system_events "
                    "(event_type, timestamp, face_id, track_id, frame_idx, metadata) "
                    "VALUES (?,?,?,?,?,?)",
                    (event_type, ts.isoformat(timespec="milliseconds"), face_id, track_id,
                     frame_idx, json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"))),
                )
                self.conn.commit()
                return int(cur.lastrowid)
            except Exception:
                self.conn.rollback()
                raise

    def audit_events_for_recovery(self) -> list:
        """Return durable audit rows so missing text-log lines can be restored."""
        with self.lock:
            return self.conn.execute(
                "SELECT audit_id, event_type, timestamp, face_id, track_id, frame_idx, image_path, metadata "
                "FROM system_events ORDER BY audit_id"
            ).fetchall()

    def events_for_recovery(self) -> list:
        """Return committed events so the text log can recover after a crash."""
        with self.lock:
            return self.conn.execute(
                "SELECT event_id, face_id, event_type, timestamp, image_path, track_id, frame_idx "
                "FROM events ORDER BY event_id"
            ).fetchall()

    # ---- queries ----
    def unique_count(self) -> int:
        with self.lock:
            return self.conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]

    def event_counts(self) -> dict:
        with self.lock:
            rows = self.conn.execute(
                "SELECT event_type, COUNT(*) FROM events GROUP BY event_type").fetchall()
        return {k: v for k, v in rows}

    def close(self) -> None:
        with self.lock:
            self.conn.commit()
            self.conn.close()
