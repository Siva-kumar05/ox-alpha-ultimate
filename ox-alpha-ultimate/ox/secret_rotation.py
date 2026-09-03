"""Secret Rotation Mechanism."""
from __future__ import annotations
import os
import threading
import time
import secrets
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from .core import LOG, iso


class SecretType(Enum):
    API_KEY = "api_key"
    API_SECRET = "api_secret"
    TOKEN = "token"
    PASSWORD = "password"
    CERTIFICATE = "certificate"
    ENCRYPTION_KEY = "encryption_key"


@dataclass
class Secret:
    """Secret with metadata."""
    name: str
    secret_type: SecretType
    value: str
    created_at: str
    expires_at: Optional[str] = None
    rotation_interval_days: int = 90
    version: int = 1
    active: bool = True
    metadata: Dict = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        return datetime.fromisoformat(self.expires_at) < datetime.now()
    
    def days_until_expiry(self) -> Optional[int]:
        if not self.expires_at:
            return None
        delta = datetime.fromisoformat(self.expires_at) - datetime.now()
        return delta.days


@dataclass
class RotationPolicy:
    """Secret rotation policy."""
    secret_type: SecretType
    interval_days: int
    warning_days: int = 7
    auto_rotate: bool = True
    generator: Optional[Callable[[], str]] = None
    validators: List[Callable[[str], bool]] = field(default_factory=list)


class SecretManager:
    """Manages secrets with rotation."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.secret_cfg = cfg.get("secret_rotation", {})
        self.enabled = self.secret_cfg.get("enabled", True)
        self.check_interval_hours = self.secret_cfg.get("check_interval_hours", 24)
        self.secrets_dir = Path(self.secret_cfg.get("secrets_dir", ".secrets"))
        self.encryption_key_env = self.secret_cfg.get("encryption_key_env", "SECRET_ENCRYPTION_KEY")
        
        self._secrets: Dict[str, Secret] = {}
        self._policies: Dict[SecretType, RotationPolicy] = {}
        self._rotation_callbacks: List[Callable[[Secret], bool]] = []
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Create secrets directory
        self.secrets_dir.mkdir(parents=True, exist_ok=True)
        
        # Load secrets
        self._load_secrets()
    
    def _load_secrets(self):
        """Load secrets from environment and files."""
        # Load from environment variables
        env_mapping = self.secret_cfg.get("env_mapping", {})
        for secret_name, env_var in env_mapping.items():
            value = os.getenv(env_var)
            if value:
                self._secrets[secret_name] = Secret(
                    name=secret_name,
                    secret_type=SecretType.API_KEY,
                    value=value,
                    created_at=iso(),
                    metadata={"source": "environment", "env_var": env_var}
                )
        
        # Load from files
        for secret_file in self.secrets_dir.glob("*.secret"):
            try:
                name = secret_file.stem
                value = secret_file.read_text().strip()
                if value:
                    self._secrets[name] = Secret(
                        name=name,
                        secret_type=SecretType.API_SECRET,
                        value=value,
                        created_at=iso(),
                        metadata={"source": "file", "path": str(secret_file)}
                    )
            except Exception as e:
                LOG.error(f"Failed to load secret {secret_file}: {e}")
    
    def set_policy(self, policy: RotationPolicy):
        """Set rotation policy for secret type."""
        with self._lock:
            self._policies[policy.secret_type] = policy
    
    def add_rotation_callback(self, callback: Callable[[Secret], bool]):
        """Add callback for secret rotation."""
        self._rotation_callbacks.append(callback)
    
    def get_secret(self, name: str) -> Optional[str]:
        """Get secret value."""
        with self._lock:
            secret = self._secrets.get(name)
            if secret and secret.active:
                return secret.value
        return None
    
    def get_secret_obj(self, name: str) -> Optional[Secret]:
        """Get secret object."""
        with self._lock:
            return self._secrets.get(name)
    
    def store_secret(
        self,
        name: str,
        value: str,
        secret_type: SecretType = SecretType.API_KEY,
        rotation_interval_days: int = 90
    ):
        """Store a new secret."""
        with self._lock:
            expires_at = (datetime.now() + timedelta(days=rotation_interval_days)).isoformat()
            
            secret = Secret(
                name=name,
                secret_type=secret_type,
                value=value,
                created_at=iso(),
                expires_at=expires_at,
                rotation_interval_days=rotation_interval_days,
                version=1
            )
            
            self._secrets[name] = secret
            self._persist_secret(secret)
    
    def rotate_secret(self, name: str) -> bool:
        """Rotate a secret."""
        with self._lock:
            old_secret = self._secrets.get(name)
            if not old_secret:
                LOG.error(f"Secret not found: {name}")
                return False
            
            # Generate new value
            policy = self._policies.get(old_secret.secret_type)
            if policy and policy.generator:
                new_value = policy.generator()
            else:
                new_value = self._generate_secret(old_secret.secret_type)
            
            # Validate
            if policy:
                for validator in policy.validators:
                    if not validator(new_value):
                        LOG.error(f"Generated secret failed validation: {name}")
                        return False
            
            # Create new secret
            new_secret = Secret(
                name=name,
                secret_type=old_secret.secret_type,
                value=new_value,
                created_at=iso(),
                expires_at=(datetime.now() + timedelta(days=old_secret.rotation_interval_days)).isoformat(),
                rotation_interval_days=old_secret.rotation_interval_days,
                version=old_secret.version + 1,
                metadata={**old_secret.metadata, "previous_version": old_secret.version}
            )
            
            self._secrets[name] = new_secret
            self._persist_secret(new_secret)
            
            # Deactivate old
            old_secret.active = False
            old_secret.metadata["rotated_at"] = iso()
            old_secret.metadata["rotated_to_version"] = new_secret.version
            
            LOG.info(f"Rotated secret: {name} (v{old_secret.version} -> v{new_secret.version})")
            
            # Call rotation callbacks
            for callback in self._rotation_callbacks:
                try:
                    callback(new_secret)
                except Exception as e:
                    LOG.error(f"Rotation callback failed: {e}")
            
            return True
    
    def _generate_secret(self, secret_type: SecretType) -> str:
        """Generate a new secret value."""
        if secret_type == SecretType.API_KEY:
            return secrets.token_urlsafe(32)
        elif secret_type == SecretType.API_SECRET:
            return secrets.token_urlsafe(48)
        elif secret_type == SecretType.TOKEN:
            return secrets.token_urlsafe(32)
        elif secret_type == SecretType.PASSWORD:
            return secrets.token_urlsafe(24)
        elif secret_type == SecretType.ENCRYPTION_KEY:
            return base64.b64encode(secrets.token_bytes(32)).decode()
        else:
            return secrets.token_urlsafe(32)
    
    def _persist_secret(self, secret: Secret):
        """Persist secret to file."""
        secret_file = self.secrets_dir / f"{secret.name}.secret"
        try:
            secret_file.write_text(secret.value)
            secret_file.chmod(0o600)
        except Exception as e:
            LOG.error(f"Failed to persist secret {secret.name}: {e}")
    
    def start_auto_rotation(self):
        """Start automatic secret rotation."""
        if not self.enabled:
            return
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._rotation_loop, daemon=True)
        self._thread.start()
        LOG.info("Secret auto-rotation started")
    
    def stop_auto_rotation(self):
        """Stop automatic rotation."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        LOG.info("Secret auto-rotation stopped")
    
    def _rotation_loop(self):
        """Main rotation loop."""
        while self._running:
            try:
                self._check_and_rotate()
            except Exception as e:
                LOG.error(f"Rotation check failed: {e}")
            
            # Sleep in intervals
            for _ in range(self.check_interval_hours * 60):
                if not self._running:
                    break
                time.sleep(60)
    
    def _check_and_rotate(self):
        """Check and rotate expired secrets."""
        with self._lock:
            for name, secret in self._secrets.items():
                if not secret.active:
                    continue
                
                policy = self._policies.get(secret.secret_type)
                if not policy or not policy.auto_rotate:
                    continue
                
                if secret.is_expired():
                    LOG.warning(f"Secret expired, rotating: {name}")
                    self.rotate_secret(name)
                elif secret.days_until_expiry() is not None and secret.days_until_expiry() <= policy.warning_days:
                    LOG.warning(f"Secret expiring soon: {name} ({secret.days_until_expiry()} days)")
    
    def get_status(self) -> Dict:
        """Get secret manager status."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "auto_rotation": self._running,
                "total_secrets": len(self._secrets),
                "active_secrets": sum(1 for s in self._secrets.values() if s.active),
                "expired_secrets": sum(1 for s in self._secrets.values() if s.is_expired()),
                "expiring_soon": sum(
                    1 for s in self._secrets.values() 
                    if s.days_until_expiry() is not None and s.days_until_expiry() <= 7
                ),
                "secrets": {
                    name: {
                        "type": s.secret_type.value,
                        "version": s.version,
                        "active": s.active,
                        "created_at": s.created_at,
                        "expires_at": s.expires_at,
                        "days_until_expiry": s.days_until_expiry()
                    }
                    for name, s in self._secrets.items()
                }
            }


# Default generators
def generate_dhan_token() -> str:
    """Generate Dhan token (placeholder - would use actual OAuth)."""
    return secrets.token_urlsafe(64)


def generate_groww_secret() -> str:
    """Generate Groww secret."""
    return secrets.token_urlsafe(48)


def create_default_secret_manager(cfg) -> SecretManager:
    """Create secret manager with default policies."""
    manager = SecretManager(cfg)
    
    # Set default policies
    manager.set_policy(RotationPolicy(
        secret_type=SecretType.API_KEY,
        interval_days=90,
        warning_days=14,
        auto_rotate=True,
        generator=generate_dhan_token
    ))
    
    manager.set_policy(RotationPolicy(
        secret_type=SecretType.API_SECRET,
        interval_days=90,
        warning_days=14,
        auto_rotate=True,
        generator=generate_groww_secret
    ))
    
    manager.set_policy(RotationPolicy(
        secret_type=SecretType.TOKEN,
        interval_days=30,
        warning_days=7,
        auto_rotate=True
    ))
    
    manager.set_policy(RotationPolicy(
        secret_type=SecretType.ENCRYPTION_KEY,
        interval_days=365,
        warning_days=30,
        auto_rotate=False  # Manual rotation for encryption keys
    ))
    
    return manager