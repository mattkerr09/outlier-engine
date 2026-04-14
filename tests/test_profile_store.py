"""
Tests for outlier_engine.profile_store — SQLite WAL profile storage.

Covers:
  - 100 save/load cycles
  - Concurrent writes (threading)
  - Migration from legacy JSON
  - Backup creation and pruning
  - GDPR export
  - delete_all
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import pytest

from outlier_engine.profile_store import ProfileStore


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture()
def tmp_data_dir(tmp_path: Path) -> Path:
    """Return a fresh temporary data directory for each test."""
    d = tmp_path / "outlier_data"
    d.mkdir()
    return d


@pytest.fixture()
def store(tmp_data_dir: Path) -> ProfileStore:
    """Return a ProfileStore backed by a temp directory."""
    return ProfileStore(data_dir=tmp_data_dir)


# ------------------------------------------------------------------
# Basic CRUD
# ------------------------------------------------------------------

class TestBasicCRUD:
    def test_save_and_load(self, store: ProfileStore) -> None:
        store.save("p1", "Profile One", [0.1, 0.2, 0.3])
        loaded = store.load("p1")
        assert loaded is not None
        assert loaded["name"] == "Profile One"
        assert loaded["alphas"] == [0.1, 0.2, 0.3]
        assert loaded["source"] == "user"
        assert loaded["user_edits"] == 0

    def test_load_missing_returns_none(self, store: ProfileStore) -> None:
        assert store.load("nonexistent") is None

    def test_update_increments_edits(self, store: ProfileStore) -> None:
        store.save("p1", "V1", [1.0])
        store.save("p1", "V2", [2.0])
        loaded = store.load("p1")
        assert loaded is not None
        assert loaded["name"] == "V2"
        assert loaded["user_edits"] == 1

    def test_delete(self, store: ProfileStore) -> None:
        store.save("p1", "X", [0.0])
        assert store.delete("p1") is True
        assert store.load("p1") is None
        assert store.delete("p1") is False

    def test_delete_all(self, store: ProfileStore) -> None:
        for i in range(5):
            store.save(f"p{i}", f"Name{i}", [float(i)])
        assert store.count() == 5
        deleted = store.delete_all()
        assert deleted == 5
        assert store.count() == 0

    def test_load_all(self, store: ProfileStore) -> None:
        store.save("a", "Alpha", [1.0])
        store.save("b", "Beta", [2.0])
        all_profiles = store.load_all()
        assert set(all_profiles.keys()) == {"a", "b"}

    def test_count(self, store: ProfileStore) -> None:
        assert store.count() == 0
        store.save("x", "X", [])
        assert store.count() == 1


# ------------------------------------------------------------------
# 100 save/load cycles
# ------------------------------------------------------------------

class TestSaveLoadCycles:
    def test_100_save_load_cycles(self, store: ProfileStore) -> None:
        """Write and read back 100 distinct profiles, then verify all."""
        for i in range(100):
            pid = f"profile_{i:04d}"
            alphas = [float(i) / 100.0] * 10
            store.save(pid, f"Name {i}", alphas, source="cycle_test")

        assert store.count() == 100

        for i in range(100):
            pid = f"profile_{i:04d}"
            loaded = store.load(pid)
            assert loaded is not None, f"Profile {pid} missing"
            assert loaded["name"] == f"Name {i}"
            expected = [float(i) / 100.0] * 10
            assert loaded["alphas"] == expected
            assert loaded["source"] == "cycle_test"

    def test_100_update_cycles_on_single_profile(self, store: ProfileStore) -> None:
        """Update the same profile 100 times; verify final state and edit count."""
        store.save("evolving", "V0", [0.0])
        for i in range(1, 101):
            store.save("evolving", f"V{i}", [float(i)])

        loaded = store.load("evolving")
        assert loaded is not None
        assert loaded["name"] == "V100"
        assert loaded["alphas"] == [100.0]
        assert loaded["user_edits"] == 100


# ------------------------------------------------------------------
# Concurrent writes (threading)
# ------------------------------------------------------------------

class TestConcurrentWrites:
    def test_concurrent_saves(self, tmp_data_dir: Path) -> None:
        """Spawn 10 threads each writing 20 profiles — all must survive."""
        num_threads = 10
        writes_per_thread = 20
        errors: list = []

        def writer(thread_id: int) -> None:
            try:
                # Each thread gets its own store instance (and thus its own connection)
                s = ProfileStore(data_dir=tmp_data_dir)
                for j in range(writes_per_thread):
                    pid = f"t{thread_id}_p{j}"
                    s.save(pid, f"Thread{thread_id} Profile{j}", [float(thread_id)])
                s.close()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Thread errors: {errors}"

        # Verify all profiles exist
        store = ProfileStore(data_dir=tmp_data_dir)
        assert store.count() == num_threads * writes_per_thread

    def test_concurrent_read_write(self, tmp_data_dir: Path) -> None:
        """One writer thread, several reader threads running simultaneously."""
        store_w = ProfileStore(data_dir=tmp_data_dir)
        # Pre-populate
        for i in range(50):
            store_w.save(f"r{i}", f"R{i}", [float(i)])

        errors: list = []
        stop_event = threading.Event()

        def reader() -> None:
            try:
                s = ProfileStore(data_dir=tmp_data_dir)
                while not stop_event.is_set():
                    all_p = s.load_all()
                    assert isinstance(all_p, dict)
                s.close()
            except Exception as exc:
                errors.append(exc)

        def writer() -> None:
            try:
                s = ProfileStore(data_dir=tmp_data_dir)
                for i in range(50):
                    s.save(f"w{i}", f"W{i}", [float(i)])
                s.close()
            except Exception as exc:
                errors.append(exc)

        readers = [threading.Thread(target=reader) for _ in range(4)]
        w = threading.Thread(target=writer)

        for r in readers:
            r.start()
        w.start()
        w.join(timeout=30)
        stop_event.set()
        for r in readers:
            r.join(timeout=10)

        assert errors == [], f"Errors: {errors}"

        final = ProfileStore(data_dir=tmp_data_dir)
        assert final.count() == 100  # 50 pre-populated + 50 written


# ------------------------------------------------------------------
# Migration from JSON
# ------------------------------------------------------------------

class TestMigration:
    def test_migrates_legacy_json(self, tmp_data_dir: Path) -> None:
        """If profiles.json exists, it should be imported and renamed."""
        legacy = {
            "default": {
                "name": "Default",
                "color": "#888888",
                "alpha": [0.0] * 10,
            },
            "medical": {
                "name": "Medical",
                "color": "#ff6b6b",
                "alpha": [0.5] * 10,
            },
        }
        json_path = tmp_data_dir / "profiles.json"
        json_path.write_text(json.dumps(legacy))

        store = ProfileStore(data_dir=tmp_data_dir)

        # JSON should be renamed
        assert not json_path.exists()
        assert (tmp_data_dir / "profiles_legacy_backup.json").exists()

        # Profiles should be in SQLite
        assert store.count() == 2
        default = store.load("default")
        assert default is not None
        assert default["name"] == "Default"
        assert default["alphas"] == [0.0] * 10
        assert default["source"] == "migrated_json"

    def test_no_migration_when_no_json(self, tmp_data_dir: Path) -> None:
        """No crash when profiles.json doesn't exist."""
        store = ProfileStore(data_dir=tmp_data_dir)
        assert store.count() == 0

    def test_migration_is_idempotent(self, tmp_data_dir: Path) -> None:
        """Second init should not fail if backup already exists."""
        legacy = {"x": {"name": "X", "alpha": [1.0]}}
        (tmp_data_dir / "profiles.json").write_text(json.dumps(legacy))

        s1 = ProfileStore(data_dir=tmp_data_dir)
        assert s1.count() == 1

        # Second init — no JSON to migrate, should be fine
        s2 = ProfileStore(data_dir=tmp_data_dir)
        assert s2.count() == 1

    def test_migration_with_alphas_key(self, tmp_data_dir: Path) -> None:
        """Migration handles both 'alpha' and 'alphas' keys."""
        legacy = {
            "new_format": {
                "name": "NewFormat",
                "alphas": [0.9, 0.8],
            },
        }
        (tmp_data_dir / "profiles.json").write_text(json.dumps(legacy))
        store = ProfileStore(data_dir=tmp_data_dir)
        loaded = store.load("new_format")
        assert loaded is not None
        assert loaded["alphas"] == [0.9, 0.8]


# ------------------------------------------------------------------
# Backup creation and pruning
# ------------------------------------------------------------------

class TestBackups:
    def test_daily_backup_created(self, tmp_data_dir: Path) -> None:
        """Initialising the store should create today's backup."""
        store = ProfileStore(data_dir=tmp_data_dir)
        today_str = date.today().strftime("%Y-%m-%d")
        backup_file = tmp_data_dir / "backups" / f"profiles-{today_str}.db"
        assert backup_file.exists()

    def test_backup_not_duplicated(self, tmp_data_dir: Path) -> None:
        """Second init on the same day should not overwrite the backup."""
        s1 = ProfileStore(data_dir=tmp_data_dir)
        today_str = date.today().strftime("%Y-%m-%d")
        backup_file = tmp_data_dir / "backups" / f"profiles-{today_str}.db"
        mtime1 = backup_file.stat().st_mtime

        # Tiny delay so mtime would differ if rewritten
        time.sleep(0.05)

        s2 = ProfileStore(data_dir=tmp_data_dir)
        mtime2 = backup_file.stat().st_mtime
        assert mtime1 == mtime2

    def test_prune_old_backups(self, tmp_data_dir: Path) -> None:
        """Backups older than 30 days should be pruned."""
        backup_dir = tmp_data_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Create some fake old backups
        old_dates = [
            date.today() - timedelta(days=d) for d in [31, 40, 60, 100]
        ]
        recent_dates = [
            date.today() - timedelta(days=d) for d in [0, 5, 15, 29]
        ]

        for d in old_dates + recent_dates:
            f = backup_dir / f"profiles-{d.strftime('%Y-%m-%d')}.db"
            f.write_text("fake")

        # Constructing a store triggers pruning
        store = ProfileStore(data_dir=tmp_data_dir)

        remaining = sorted(f.name for f in backup_dir.iterdir())
        for d in old_dates:
            fname = f"profiles-{d.strftime('%Y-%m-%d')}.db"
            assert fname not in remaining, f"{fname} should have been pruned"

        for d in recent_dates:
            fname = f"profiles-{d.strftime('%Y-%m-%d')}.db"
            assert fname in remaining, f"{fname} should have been kept"


# ------------------------------------------------------------------
# GDPR export
# ------------------------------------------------------------------

class TestGDPR:
    def test_export_gdpr(self, store: ProfileStore) -> None:
        store.save("a", "Alpha", [1.0, 2.0], source="test")
        store.save("b", "Beta", [3.0], source="test")

        export = store.export_gdpr()
        assert export["export_type"] == "gdpr_full"
        assert export["profile_count"] == 2
        assert "a" in export["profiles"]
        assert "b" in export["profiles"]
        # Must be JSON-serialisable
        json.dumps(export)

    def test_export_empty(self, store: ProfileStore) -> None:
        export = store.export_gdpr()
        assert export["profile_count"] == 0
        assert export["profiles"] == {}


# ------------------------------------------------------------------
# WAL mode verification
# ------------------------------------------------------------------

class TestWALMode:
    def test_wal_mode_enabled(self, tmp_data_dir: Path) -> None:
        store = ProfileStore(data_dir=tmp_data_dir)
        conn = store._get_conn()
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode == "wal"


# ------------------------------------------------------------------
# Atomicity sanity check
# ------------------------------------------------------------------

class TestAtomicity:
    def test_failed_save_does_not_corrupt(self, tmp_data_dir: Path) -> None:
        """If a save raises mid-transaction, the db should be unchanged."""
        store = ProfileStore(data_dir=tmp_data_dir)
        store.save("ok", "OK", [1.0])

        # Patch json.dumps to explode only for the "bad" profile's alphas,
        # which is called inside the transaction after BEGIN IMMEDIATE.
        original_dumps = json.dumps
        explode = False

        def patched_dumps(obj, *a, **kw):
            if explode and isinstance(obj, list) and obj == [9.9]:
                raise RuntimeError("injected failure")
            return original_dumps(obj, *a, **kw)

        import outlier_engine.profile_store as ps_mod
        old = ps_mod.json.dumps
        ps_mod.json.dumps = patched_dumps
        explode = True

        with pytest.raises(RuntimeError, match="injected"):
            store.save("bad", "BAD", [9.9])

        ps_mod.json.dumps = old

        # "ok" should still be there, "bad" should not
        assert store.load("ok") is not None
        assert store.load("bad") is None
