"""
Atomic profile storage via SQLite WAL mode.

Replaces the raw JSON profile storage (~/.outlier/profiles.json) with a
durable SQLite database that uses WAL journaling for concurrent-read safety
and BEGIN IMMEDIATE transactions for write atomicity.

On first use, any existing profiles.json is migrated into the database and
renamed to profiles_legacy_backup.json.

Daily backups are created in ~/.outlier/backups/ and pruned to 30 days.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "ProfileStore",
    "get_default_store",
]

_SCHEMA_VERSION = 1

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS profiles (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    alphas      TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL,
    source      TEXT NOT NULL DEFAULT 'user',
    user_edits  INTEGER NOT NULL DEFAULT 0
);
"""

_CREATE_META = """\
CREATE TABLE IF NOT EXISTS _meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class ProfileStore:
    """Thread-safe, WAL-mode SQLite profile store."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else Path.home() / ".outlier"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "profiles.db"
        self.backup_dir = self.data_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._init_lock = threading.Lock()

        # Ensure schema exists (first connection)
        conn = self._get_conn()
        self._ensure_schema(conn)

        # Migrate legacy JSON if present
        self._migrate_legacy_json()

        # Daily backup + prune
        self._daily_backup()
        self._prune_backups(keep_days=30)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Return a per-thread connection with WAL mode enabled."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=10,
                check_same_thread=False,
            )
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        with self._init_lock:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_META)
            conn.execute(
                "INSERT OR IGNORE INTO _meta (key, value) VALUES (?, ?)",
                ("schema_version", str(_SCHEMA_VERSION)),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Legacy migration
    # ------------------------------------------------------------------

    def _migrate_legacy_json(self) -> None:
        json_path = self.data_dir / "profiles.json"
        if not json_path.exists():
            return

        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        if not isinstance(raw, dict) or len(raw) == 0:
            return

        conn = self._get_conn()
        now_ts = int(time.time())
        conn.execute("BEGIN IMMEDIATE;")
        try:
            for pid, pdata in raw.items():
                # Normalise: the server.py format uses "alpha" (list),
                # our schema stores "alphas" as JSON text.
                alphas = pdata.get("alphas") or pdata.get("alpha", [])
                name = pdata.get("name", pid)
                source = pdata.get("source", "migrated_json")
                user_edits = int(pdata.get("user_edits", 0))
                created_at = int(pdata.get("created_at", now_ts))
                updated_at = int(pdata.get("updated_at", now_ts))

                conn.execute(
                    """INSERT OR IGNORE INTO profiles
                       (id, name, alphas, created_at, updated_at, source, user_edits)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(pid),
                        name,
                        json.dumps(alphas),
                        created_at,
                        updated_at,
                        source,
                        user_edits,
                    ),
                )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise

        # Rename legacy file
        backup_path = self.data_dir / "profiles_legacy_backup.json"
        json_path.rename(backup_path)

    # ------------------------------------------------------------------
    # Daily backup & pruning
    # ------------------------------------------------------------------

    def _daily_backup(self) -> None:
        today_str = date.today().strftime("%Y-%m-%d")
        backup_file = self.backup_dir / f"profiles-{today_str}.db"
        if backup_file.exists():
            return
        if not self.db_path.exists():
            return
        shutil.copy2(str(self.db_path), str(backup_file))

    def _prune_backups(self, keep_days: int = 30) -> None:
        cutoff = date.today() - timedelta(days=keep_days)
        for f in self.backup_dir.iterdir():
            if not f.name.startswith("profiles-") or not f.name.endswith(".db"):
                continue
            # Extract date from filename: profiles-YYYY-MM-DD.db
            try:
                date_str = f.name[len("profiles-"):-len(".db")]
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if file_date < cutoff:
                f.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def save(
        self,
        profile_id: str,
        name: str,
        alphas: list,
        source: str = "user",
        user_edits: Optional[int] = None,
    ) -> None:
        """Insert or update a profile atomically."""
        conn = self._get_conn()
        now_ts = int(time.time())
        conn.execute("BEGIN IMMEDIATE;")
        try:
            existing = conn.execute(
                "SELECT created_at, user_edits FROM profiles WHERE id = ?",
                (profile_id,),
            ).fetchone()

            if existing:
                created = existing["created_at"]
                edits = (user_edits if user_edits is not None
                         else existing["user_edits"] + 1)
                conn.execute(
                    """UPDATE profiles
                       SET name=?, alphas=?, updated_at=?, source=?, user_edits=?
                       WHERE id=?""",
                    (name, json.dumps(alphas), now_ts, source, edits, profile_id),
                )
            else:
                edits = user_edits if user_edits is not None else 0
                conn.execute(
                    """INSERT INTO profiles
                       (id, name, alphas, created_at, updated_at, source, user_edits)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (profile_id, name, json.dumps(alphas), now_ts, now_ts, source, edits),
                )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    def load(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Load a single profile by id. Returns None if not found."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """Load all profiles as {id: {...}}."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM profiles ORDER BY name").fetchall()
        return {row["id"]: self._row_to_dict(row) for row in rows}

    def delete(self, profile_id: str) -> bool:
        """Delete a single profile. Returns True if it existed."""
        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            cur = conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
            conn.execute("COMMIT;")
            return cur.rowcount > 0
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    def delete_all(self) -> int:
        """Delete ALL profile data. Returns count of deleted rows."""
        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            cur = conn.execute("DELETE FROM profiles")
            conn.execute("COMMIT;")
            return cur.rowcount
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    def count(self) -> int:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) AS n FROM profiles").fetchone()
        return row["n"]

    # ------------------------------------------------------------------
    # GDPR
    # ------------------------------------------------------------------

    def export_gdpr(self) -> Dict[str, Any]:
        """Return all profile data as a JSON-serialisable dict (GDPR export)."""
        profiles = self.load_all()
        return {
            "export_type": "gdpr_full",
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "profile_count": len(profiles),
            "profiles": profiles,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["alphas"] = json.loads(d["alphas"])
        return d

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_default_store: Optional[ProfileStore] = None
_default_lock = threading.Lock()


def get_default_store(data_dir: Optional[Path] = None) -> ProfileStore:
    """Return (or create) the module-level default ProfileStore."""
    global _default_store
    with _default_lock:
        if _default_store is None:
            _default_store = ProfileStore(data_dir=data_dir)
        return _default_store
