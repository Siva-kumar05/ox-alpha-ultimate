"""Database Backup Strategy."""
from __future__ import annotations
import sqlite3
import threading
import time
import shutil
import gzip
import os
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta
from .core import LOG, iso


@dataclass
class BackupInfo:
    """Backup metadata."""
    backup_id: str
    source_path: str
    backup_path: str
    size_bytes: int
    compressed: bool
    timestamp: str
    duration_seconds: float
    status: str  # "success", "failed", "in_progress"
    error: Optional[str] = None


class DatabaseBackupManager:
    """Manages database backups with rotation and compression."""
    
    def __init__(self, cfg, db_path: str):
        self.cfg = cfg
        self.db_path = Path(db_path).resolve()
        self.backup_cfg = cfg.get("database_backup", {})
        self.enabled = self.backup_cfg.get("enabled", True)
        self.backup_dir = Path(self.backup_cfg.get("backup_dir", "backups")).resolve()
        self.interval_hours = self.backup_cfg.get("interval_hours", 6)
        self.retention_days = self.backup_cfg.get("retention_days", 30)
        self.max_backups = self.backup_cfg.get("max_backups", 100)
        self.compress = self.backup_cfg.get("compress", True)
        self.backup_timeout = self.backup_cfg.get("timeout_seconds", 300)
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._last_backup: Optional[BackupInfo] = None
        self._backup_history: List[BackupInfo] = []
        
        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Load history
        self._load_history()
    
    def _load_history(self):
        """Load backup history from metadata file."""
        meta_file = self.backup_dir / "backup_history.json"
        if meta_file.exists():
            try:
                import json
                with open(meta_file) as f:
                    data = json.load(f)
                    self._backup_history = [
                        BackupInfo(**item) for item in data.get("backups", [])
                    ]
            except Exception as e:
                LOG.warning(f"Failed to load backup history: {e}")
    
    def _save_history(self):
        """Save backup history to metadata file."""
        meta_file = self.backup_dir / "backup_history.json"
        try:
            import json
            with open(meta_file, "w") as f:
                json.dump({
                    "backups": [
                        {
                            "backup_id": b.backup_id,
                            "source_path": b.source_path,
                            "backup_path": b.backup_path,
                            "size_bytes": b.size_bytes,
                            "compressed": b.compressed,
                            "timestamp": b.timestamp,
                            "duration_seconds": b.duration_seconds,
                            "status": b.status,
                            "error": b.error
                        }
                        for b in self._backup_history
                    ]
                }, f, indent=2)
        except Exception as e:
            LOG.error(f"Failed to save backup history: {e}")
    
    def start(self):
        """Start automatic backups."""
        if not self.enabled:
            return
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._backup_loop, daemon=True)
        self._thread.start()
        LOG.info(f"Database backup manager started (interval: {self.interval_hours}h)")
    
    def stop(self):
        """Stop automatic backups."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        LOG.info("Database backup manager stopped")
    
    def _backup_loop(self):
        """Main backup loop."""
        while self._running:
            try:
                # Check if backup is due
                if self._should_backup():
                    self.create_backup()
            except Exception as e:
                LOG.error(f"Backup loop error: {e}")
            
            # Sleep in small intervals to allow quick shutdown
            for _ in range(int(self.interval_hours * 3600 / 60)):
                if not self._running:
                    break
                time.sleep(60)
    
    def _should_backup(self) -> bool:
        """Check if backup is due."""
        if not self._last_backup:
            return True
        
        last_time = datetime.fromisoformat(self._last_backup.timestamp)
        next_due = last_time + timedelta(hours=self.interval_hours)
        return datetime.now() >= next_due
    
    def create_backup(self, force: bool = False) -> BackupInfo:
        """Create a database backup."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if self.compress:
            backup_path = self.backup_dir / f"{backup_id}.db.gz"
        else:
            backup_path = self.backup_dir / f"{backup_id}.db"
        
        info = BackupInfo(
            backup_id=backup_id,
            source_path=str(self.db_path),
            backup_path=str(backup_path),
            size_bytes=0,
            compressed=self.compress,
            timestamp=iso(),
            duration_seconds=0.0,
            status="in_progress"
        )
        
        start_time = time.time()
        
        try:
            # Use SQLite backup API for consistent backup
            if self.compress:
                self._create_compressed_backup(backup_path)
            else:
                self._create_simple_backup(backup_path)
            
            # Verify backup
            if not self._verify_backup(backup_path):
                raise RuntimeError("Backup verification failed")
            
            info.size_bytes = backup_path.stat().st_size
            info.duration_seconds = time.time() - start_time
            info.status = "success"
            
            LOG.info(f"Backup created: {backup_path} ({info.size_bytes/1e6:.1f}MB in {info.duration_seconds:.1f}s)")
            
        except Exception as e:
            info.duration_seconds = time.time() - start_time
            info.status = "failed"
            info.error = str(e)
            LOG.error(f"Backup failed: {e}")
            
            # Clean up failed backup
            if backup_path.exists():
                backup_path.unlink()
        
        with self._lock:
            self._last_backup = info
            self._backup_history.append(info)
            self._cleanup_old_backups()
            self._save_history()
        
        return info
    
    def _create_simple_backup(self, backup_path: Path):
        """Create simple file copy backup."""
        # Use SQLite backup API for consistency
        source = sqlite3.connect(self.db_path)
        dest = sqlite3.connect(backup_path)
        
        try:
            source.backup(dest)
        finally:
            source.close()
            dest.close()
    
    def _create_compressed_backup(self, backup_path: Path):
        """Create compressed backup."""
        # First create temp uncompressed backup
        temp_path = backup_path.with_suffix(".tmp.db")
        
        try:
            source = sqlite3.connect(self.db_path)
            dest = sqlite3.connect(temp_path)
            
            try:
                source.backup(dest)
            finally:
                source.close()
                dest.close()
            
            # Compress
            with open(temp_path, "rb") as f_in:
                with gzip.open(backup_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def _verify_backup(self, backup_path: Path) -> bool:
        """Verify backup integrity."""
        try:
            if self.compress:
                # Decompress to temp and verify
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                    temp_path = Path(tmp.name)
                
                try:
                    with gzip.open(backup_path, "rb") as f_in:
                        with open(temp_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    conn = sqlite3.connect(temp_path)
                    conn.execute("PRAGMA integrity_check")
                    conn.close()
                    return True
                finally:
                    if temp_path.exists():
                        temp_path.unlink()
            else:
                conn = sqlite3.connect(backup_path)
                conn.execute("PRAGMA integrity_check")
                conn.close()
                return True
        except Exception as e:
            LOG.error(f"Backup verification failed: {e}")
            return False
    
    def _cleanup_old_backups(self):
        """Clean up old backups based on retention policy."""
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        
        # Remove by age
        to_remove = []
        for backup in self._backup_history:
            backup_time = datetime.fromisoformat(backup.timestamp)
            if backup_time < cutoff:
                to_remove.append(backup)
        
        # Remove by count
        if len(self._backup_history) > self.max_backups:
            # Sort by timestamp, remove oldest
            sorted_backups = sorted(self._backup_history, key=lambda b: b.timestamp)
            to_remove.extend(sorted_backups[:len(sorted_backups) - self.max_backups])
        
        for backup in to_remove:
            try:
                backup_file = Path(backup.backup_path)
                if backup_file.exists():
                    backup_file.unlink()
                self._backup_history.remove(backup)
                LOG.info(f"Removed old backup: {backup.backup_id}")
            except Exception as e:
                LOG.error(f"Failed to remove backup {backup.backup_id}: {e}")
    
    def list_backups(self) -> List[BackupInfo]:
        """List all backups."""
        with self._lock:
            return list(self._backup_history)
    
    def restore_backup(self, backup_id: str, target_path: Optional[str] = None) -> bool:
        """Restore from backup."""
        backup = None
        for b in self._backup_history:
            if b.backup_id == backup_id:
                backup = b
                break
        
        if not backup:
            LOG.error(f"Backup not found: {backup_id}")
            return False
        
        backup_file = Path(backup.backup_path)
        if not backup_file.exists():
            LOG.error(f"Backup file not found: {backup_file}")
            return False
        
        target = Path(target_path) if target_path else self.db_path
        
        try:
            if backup.compressed:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                    temp_path = Path(tmp.name)
                
                try:
                    with gzip.open(backup_file, "rb") as f_in:
                        with open(temp_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Verify
                    conn = sqlite3.connect(temp_path)
                    conn.execute("PRAGMA integrity_check")
                    conn.close()
                    
                    # Replace target
                    if target.exists():
                        target.unlink()
                    shutil.move(temp_path, target)
                    
                    LOG.info(f"Restored backup {backup_id} to {target}")
                    return True
                finally:
                    if temp_path.exists():
                        temp_path.unlink()
            else:
                # Simple copy
                if target.exists():
                    target.unlink()
                shutil.copy2(backup_file, target)
                LOG.info(f"Restored backup {backup_id} to {target}")
                return True
        except Exception as e:
            LOG.error(f"Restore failed: {e}")
            return False
    
    def get_status(self) -> Dict:
        """Get backup manager status."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "running": self._running,
                "last_backup": {
                    "backup_id": self._last_backup.backup_id,
                    "timestamp": self._last_backup.timestamp,
                    "size_mb": round(self._last_backup.size_bytes / 1e6, 2),
                    "duration_seconds": self._last_backup.duration_seconds,
                    "status": self._last_backup.status
                } if self._last_backup else None,
                "total_backups": len(self._backup_history),
                "backup_dir": str(self.backup_dir),
                "next_backup_due": (
                    datetime.fromisoformat(self._last_backup.timestamp) + timedelta(hours=self.interval_hours)
                ).isoformat() if self._last_backup else "now"
            }