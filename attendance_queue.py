"""
SQLite-backed attendance outbox with async Zoho sync.

Flow per verify request:
  1. Face matched → is_already_marked() check (in-memory O(1), then SQLite <1ms)
  2. enqueue() → write to SQLite instantly → return success to student
  3. Background worker drains PENDING rows → posts to Zoho → marks POSTED
  4. On Zoho failure: exponential backoff retry (5s, 15s, 45s, 135s, 405s)
  5. After 5 failed attempts: mark FAILED — visible at /admin/sync-status

This removes both Zoho API calls (duplicate check + post) from the hot path,
cutting per-student latency from ~6s to ~1.5s and eliminating silent data loss
when Zoho is temporarily unreachable.
"""

import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get(
    "ATTENDANCE_DB_PATH",
    os.path.join(os.path.dirname(__file__), "data", "attendance_queue.db"),
)
MAX_ATTEMPTS = 5
WORKER_POLL_INTERVAL = 2  # seconds between drain cycles


class AttendanceQueue:
    """
    Thread-safe SQLite outbox + background Zoho sync worker.

    Multiple Gunicorn workers share the same SQLite file — SQLite's WAL mode
    handles concurrent writes. In-memory dedup sets are per-process but
    fall back to the shared SQLite for cross-process correctness.
    """

    def __init__(self, zoho_api):
        self._zoho = zoho_api
        self._lock = threading.Lock()

        # In-memory fast-path dedup (per-process, rebuilt from DB on start)
        # {date_str: set_of_student_ids}
        self._global_marked: dict[str, set] = {}
        # {(date_str, session_id): set_of_student_ids}
        self._session_marked: dict[tuple, set] = {}

        self._init_db()
        self._rebuild_dedup_from_db()

        self._worker = threading.Thread(target=self._drain_loop, daemon=True)
        self._worker.start()
        logger.info("AttendanceQueue ready — background sync worker started.")

    # ── DB helpers ────────────────────────────────────────────────────────────

    @contextmanager
    def _db(self):
        """Open a DB connection. WAL mode is set once at init, not here."""
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _set_wal_mode(self):
        """
        Enable WAL journal mode for safe multi-process concurrent writes.
        WAL is persistent in the DB file — only needs to be set once ever.
        PRAGMA journal_mode=WAL requires an exclusive lock; all 4 Gunicorn
        workers race to set it at startup, so we retry with backoff instead
        of crashing.
        """
        for attempt in range(20):
            conn = None
            try:
                conn = sqlite3.connect(DB_PATH, timeout=2)
                row = conn.execute("PRAGMA journal_mode=WAL").fetchone()
                conn.close()
                if row and row[0] == "wal":
                    return   # already set — done
                break        # set to something else but no error — move on
            except sqlite3.OperationalError:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                time.sleep(0.15 * (attempt + 1))   # 0.15s, 0.30s … up to ~3s total
        else:
            logger.warning(
                "Could not set WAL journal mode after retries — "
                "falling back to default mode (safe, slightly less concurrent)."
            )

    def _init_db(self):
        self._set_wal_mode()
        with self._db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attendance_queue (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id    TEXT    NOT NULL,
                    student_name  TEXT    NOT NULL,
                    date_str      TEXT    NOT NULL,
                    session_id    TEXT    NOT NULL DEFAULT '',
                    status        TEXT    NOT NULL DEFAULT 'PENDING',
                    attempts      INTEGER NOT NULL DEFAULT 0,
                    last_error    TEXT,
                    created_at    TEXT    NOT NULL,
                    updated_at    TEXT    NOT NULL,
                    next_retry_at TEXT    NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_status_retry "
                "ON attendance_queue(status, next_retry_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_student_date "
                "ON attendance_queue(student_id, date_str, session_id)"
            )
        logger.info(f"Attendance queue DB at: {DB_PATH}")

    def _rebuild_dedup_from_db(self):
        """Populate in-memory dedup from today's active rows on startup."""
        today = datetime.now().strftime("%d-%b-%Y")
        with self._db() as conn:
            rows = conn.execute(
                "SELECT student_id, session_id FROM attendance_queue "
                "WHERE date_str=? AND status IN ('PENDING','PROCESSING','POSTED')",
                (today,),
            ).fetchall()
        with self._lock:
            for row in rows:
                self._mark_in_memory(row["student_id"], today, row["session_id"] or None)
        logger.info(f"Dedup set: {len(rows)} students already marked today.")

    # ── Public API ────────────────────────────────────────────────────────────

    def is_already_marked(
        self,
        student_id: str,
        date_str: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """
        O(1) in-memory check, with SQLite fallback for cross-worker correctness.
        Replaces the ~2-3s Zoho duplicate-check API call.
        """
        sid = session_id or ""

        # Fast in-memory path
        with self._lock:
            if sid:
                if student_id in self._session_marked.get((date_str, sid), set()):
                    return True
            else:
                if student_id in self._global_marked.get(date_str, set()):
                    return True

        # SQLite fallback — handles another Gunicorn worker having enqueued first
        with self._db() as conn:
            if sid:
                count = conn.execute(
                    "SELECT COUNT(*) FROM attendance_queue "
                    "WHERE student_id=? AND date_str=? AND session_id=? "
                    "AND status IN ('PENDING','PROCESSING','POSTED')",
                    (student_id, date_str, sid),
                ).fetchone()[0]
            else:
                count = conn.execute(
                    "SELECT COUNT(*) FROM attendance_queue "
                    "WHERE student_id=? AND date_str=? "
                    "AND status IN ('PENDING','PROCESSING','POSTED')",
                    (student_id, date_str),
                ).fetchone()[0]

        if count > 0:
            with self._lock:
                self._mark_in_memory(student_id, date_str, session_id)
            return True
        return False

    def enqueue(
        self,
        student_id: str,
        student_name: str,
        date_str: str,
        session_id: Optional[str] = None,
    ) -> int:
        """
        Record attendance locally and queue it for Zoho sync.
        Returns the internal queue record ID.
        This call is fast (~1ms) — the student sees success immediately.
        """
        now = datetime.now().isoformat()
        sid = session_id or ""
        with self._db() as conn:
            cursor = conn.execute(
                """
                INSERT INTO attendance_queue
                    (student_id, student_name, date_str, session_id,
                     status, attempts, created_at, updated_at, next_retry_at)
                VALUES (?, ?, ?, ?, 'PENDING', 0, ?, ?, ?)
                """,
                (student_id, student_name, date_str, sid, now, now, now),
            )
            rec_id = cursor.lastrowid

        with self._lock:
            self._mark_in_memory(student_id, date_str, session_id)

        logger.info(f"Queued attendance for {student_name} (queue #{rec_id})")
        return rec_id

    def get_status_summary(self) -> dict:
        """Return queue health for /admin/sync-status."""
        since = (datetime.now() - timedelta(days=1)).strftime("%d-%b-%Y")
        with self._db() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM attendance_queue "
                "WHERE date_str >= ? GROUP BY status",
                (since,),
            ).fetchall()
            counts = {row["status"]: row["cnt"] for row in rows}

            failed = conn.execute(
                "SELECT id, student_name, date_str, session_id, attempts, last_error, created_at "
                "FROM attendance_queue WHERE status='FAILED' "
                "ORDER BY created_at DESC LIMIT 50"
            ).fetchall()

            pending_old = conn.execute(
                "SELECT id, student_name, date_str, attempts, created_at "
                "FROM attendance_queue WHERE status='PENDING' "
                "AND created_at < ? ORDER BY created_at ASC LIMIT 20",
                ((datetime.now() - timedelta(minutes=5)).isoformat(),),
            ).fetchall()

        return {
            "pending":         counts.get("PENDING", 0),
            "posted":          counts.get("POSTED",  0),
            "failed":          counts.get("FAILED",  0),
            "failed_records":  [dict(r) for r in failed],
            "stuck_pending":   [dict(r) for r in pending_old],
        }

    def retry_failed(self) -> int:
        """Reset all FAILED records to PENDING. Returns count reset."""
        now = datetime.now().isoformat()
        with self._db() as conn:
            cursor = conn.execute(
                "UPDATE attendance_queue "
                "SET status='PENDING', attempts=0, last_error=NULL, "
                "    next_retry_at=?, updated_at=? "
                "WHERE status='FAILED'",
                (now, now),
            )
            count = cursor.rowcount
        logger.info(f"Reset {count} FAILED records to PENDING.")
        return count

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _mark_in_memory(
        self, student_id: str, date_str: str, session_id: Optional[str]
    ):
        """Must be called while holding self._lock."""
        if date_str not in self._global_marked:
            self._global_marked[date_str] = set()
        self._global_marked[date_str].add(student_id)
        if session_id:
            key = (date_str, session_id)
            if key not in self._session_marked:
                self._session_marked[key] = set()
            self._session_marked[key].add(student_id)

    # ── Background drain loop ─────────────────────────────────────────────────

    def _drain_loop(self):
        """Daemon thread: continuously drains PENDING → Zoho."""
        while True:
            try:
                self._drain()
            except Exception as e:
                logger.error(f"Queue drain error: {e}")
            time.sleep(WORKER_POLL_INTERVAL)

    def _drain(self):
        now_iso = datetime.now().isoformat()

        # Reset records stuck in PROCESSING (happens if a worker crashed mid-flight)
        stale_iso = (datetime.now() - timedelta(minutes=5)).isoformat()
        with self._db() as conn:
            conn.execute(
                "UPDATE attendance_queue "
                "SET status='PENDING', next_retry_at=?, updated_at=? "
                "WHERE status='PROCESSING' AND updated_at < ?",
                (now_iso, now_iso, stale_iso),
            )

        # Find candidate PENDING rows
        with self._db() as conn:
            rows = conn.execute(
                "SELECT id, student_id, student_name, date_str, session_id, attempts "
                "FROM attendance_queue "
                "WHERE status='PENDING' AND next_retry_at <= ? "
                "ORDER BY created_at ASC LIMIT 10",
                (now_iso,),
            ).fetchall()

        for row in rows:
            rec_id = row["id"]

            # Atomically claim this record by flipping PENDING → PROCESSING.
            # SQLite serialises the UPDATE, so only one of the 4 Gunicorn workers
            # will see rowcount=1; the others get 0 and skip it — no duplicates.
            with self._db() as conn:
                cursor = conn.execute(
                    "UPDATE attendance_queue SET status='PROCESSING', updated_at=? "
                    "WHERE id=? AND status='PENDING'",
                    (now_iso, rec_id),
                )
            if cursor.rowcount == 0:
                continue   # another worker already claimed this record

            name       = row["student_name"]
            student_id = row["student_id"]
            session_id = row["session_id"] or None
            attempts   = row["attempts"]
            try:
                result = self._zoho.post_attendance(
                    student_id=student_id,
                    student_name=name,
                    verification_type="face_blink_verified",
                    session_id=session_id,
                )
                if result.get("success"):
                    self._set_posted(rec_id)
                    logger.info(f"Queue: synced {name} → Zoho (#{rec_id})")
                else:
                    self._handle_failure(
                        rec_id, attempts, result.get("error", "Zoho returned failure")
                    )
            except Exception as e:
                self._handle_failure(rec_id, attempts, str(e))

    def _set_posted(self, rec_id: int):
        now = datetime.now().isoformat()
        with self._db() as conn:
            conn.execute(
                "UPDATE attendance_queue SET status='POSTED', updated_at=? "
                "WHERE id=? AND status='PROCESSING'",
                (now, rec_id),
            )

    def _handle_failure(self, rec_id: int, attempts: int, error: str):
        now = datetime.now().isoformat()
        new_attempts = attempts + 1
        if new_attempts >= MAX_ATTEMPTS:
            with self._db() as conn:
                conn.execute(
                    "UPDATE attendance_queue "
                    "SET status='FAILED', attempts=?, last_error=?, updated_at=? "
                    "WHERE id=? AND status='PROCESSING'",
                    (new_attempts, error[:500], now, rec_id),
                )
            logger.error(
                f"Queue: #{rec_id} permanently FAILED after {MAX_ATTEMPTS} attempts: {error[:150]}"
            )
        else:
            # Exponential backoff: 5s → 15s → 45s → 135s
            delay = 5 * (3 ** attempts)
            next_retry = (datetime.now() + timedelta(seconds=delay)).isoformat()
            with self._db() as conn:
                conn.execute(
                    "UPDATE attendance_queue "
                    "SET status='PENDING', attempts=?, last_error=?, "
                    "    updated_at=?, next_retry_at=? "
                    "WHERE id=? AND status='PROCESSING'",
                    (new_attempts, error[:500], now, next_retry, rec_id),
                )
            logger.warning(
                f"Queue: #{rec_id} attempt {new_attempts}/{MAX_ATTEMPTS}, "
                f"retry in {delay}s — {error[:100]}"
            )
