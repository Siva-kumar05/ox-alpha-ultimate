"""Configuration Hot-Reload."""
from __future__ import annotations
import threading
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass
from .core import LOG, Cfg


@dataclass
class ConfigChange:
    """Configuration change event."""
    key: str
    old_value: Any
    new_value: Any
    timestamp: str
    source: str = "file"


class ConfigWatcher:
    """Watches configuration file for changes."""
    
    def __init__(self, config_path: str, cfg: Cfg, poll_interval: float = 5.0):
        self.config_path = Path(config_path).resolve()
        self.cfg = cfg
        self.poll_interval = poll_interval
        self._last_mtime = self.config_path.stat().st_mtime if self.config_path.exists() else 0
        self._callbacks: List[Callable[[Dict[str, ConfigChange]], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
    
    def add_callback(self, callback: Callable[[Dict[str, ConfigChange]], None]):
        """Add change callback."""
        with self._lock:
            self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[Dict[str, ConfigChange]], None]):
        """Remove change callback."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def start(self):
        """Start watching."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        LOG.info(f"Config watcher started for {self.config_path}")
    
    def stop(self):
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        LOG.info("Config watcher stopped")
    
    def _watch_loop(self):
        """Main watch loop."""
        while self._running:
            try:
                if self.config_path.exists():
                    current_mtime = self.config_path.stat().st_mtime
                    if current_mtime > self._last_mtime:
                        self._last_mtime = current_mtime
                        self._reload_config()
            except Exception as e:
                LOG.error(f"Config watch error: {e}")
            
            time.sleep(self.poll_interval)
    
    def _reload_config(self):
        """Reload configuration and notify callbacks."""
        try:
            # Load new config
            raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                LOG.error("Config reload failed: not a mapping")
                return
            
            # Compare with current
            changes = self._detect_changes(raw)
            
            if changes:
                # Update Cfg instance
                self.cfg.d = raw
                self.cfg._validate()  # Re-validate
                
                LOG.info(f"Configuration reloaded: {len(changes)} changes")
                
                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(changes)
                    except Exception as e:
                        LOG.error(f"Config change callback failed: {e}")
                        
        except Exception as e:
            LOG.error(f"Config reload failed: {e}")
    
    def _detect_changes(self, new_config: Dict) -> Dict[str, ConfigChange]:
        """Detect changes between old and new config."""
        from .core import iso
        changes = {}
        all_keys = set(self.cfg.d.keys()) | set(new_config.keys())
        
        for key in all_keys:
            old_val = self.cfg.d.get(key)
            new_val = new_config.get(key)
            
            if old_val != new_val:
                changes[key] = ConfigChange(
                    key=key,
                    old_value=old_val,
                    new_value=new_val,
                    timestamp=iso()
                )
        
        return changes


class HotReloadManager:
    """Manages hot-reload for the entire application."""
    
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.watcher = ConfigWatcher(cfg.path, cfg)
        self._component_callbacks: Dict[str, Callable[[Dict[str, ConfigChange]], None]] = {}
        self._lock = threading.RLock()
    
    def register_component(self, name: str, callback: Callable[[Dict[str, ConfigChange]], None]):
        """Register component for config changes."""
        with self._lock:
            self._component_callbacks[name] = callback
            self.watcher.add_callback(callback)
    
    def unregister_component(self, name: str):
        """Unregister component."""
        with self._lock:
            if name in self._component_callbacks:
                self.watcher.remove_callback(self._component_callbacks[name])
                del self._component_callbacks[name]
    
    def start(self):
        """Start hot-reload."""
        self.watcher.start()
    
    def stop(self):
        """Stop hot-reload."""
        self.watcher.stop()
    
    def force_reload(self):
        """Force configuration reload."""
        self.watcher._reload_config()


# Decorator for hot-reloadable config values
class HotConfig:
    """Descriptor for hot-reloadable config values."""
    
    def __init__(self, key: str, default: Any = None, validator: Optional[Callable[[Any], Any]] = None):
        self.key = key
        self.default = default
        self.validator = validator
        self._cached_value = default
        self._last_update = 0.0
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        # Return cached value, will be updated by callback
        return self._cached_value
    
    def update(self, value: Any):
        """Update cached value."""
        if self.validator:
            value = self.validator(value)
        self._cached_value = value
        self._last_update = time.time()


# Example usage in components:
# class MyComponent:
#     risk_per_trade = HotConfig("risk.risk_per_trade_pct", 0.5)
#     
#     def __init__(self, hot_reload: HotReloadManager):
#         hot_reload.register_component("my_component", self._on_config_change)
#     
#     def _on_config_change(self, changes):
#         if "risk.risk_per_trade_pct" in changes:
#             self.risk_per_trade.update(changes["risk.risk_per_trade_pct"].new_value)