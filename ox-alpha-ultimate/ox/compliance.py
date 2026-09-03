"""Fail-closed execution prerequisites for the autonomous agent."""

from __future__ import annotations

import ipaddress

import requests

from .core import LOG, iso, now


class Compliance:
    def __init__(self, cfg, db):
        self.cfg = cfg
        self.db = db
        self.halted = False
        self.halt_reason = ""
        self.oms = None

    def check_ip(self, override_ip: str | None = None) -> bool:
        """Verify the actual egress address; a failed lookup never passes live mode."""
        if self.cfg["mode"] == "paper" and override_ip is None:
            return True
        try:
            ip = override_ip or requests.get("https://api.ipify.org", timeout=(3.05, 5)).text.strip()
            ipaddress.ip_address(ip)
        except (requests.RequestException, ValueError) as exc:
            self.halt(f"Unable to verify egress static IP: {exc.__class__.__name__}")
            return False

        allowed = set(self.cfg.get("ip_whitelist", []))
        ok = self.cfg["mode"] == "paper" or ip in allowed
        self.db.ex("INSERT INTO events(kind,msg,ts)VALUES('IP_CHECK',?,?)", (f"egress={ip} ok={ok}", iso()))
        if not ok:
            self.halt(f"Egress IP {ip} is not in configured allowlist")
            return False
        LOG.info("Egress static-IP check passed")
        return True

    def daily_auth(self, broker) -> bool:
        """Authenticate once per session and verify it is usable before any order flow."""
        if self.cfg["mode"] == "live" and (not self.db.audit_enabled or not self.db.verify_audit()):
            self.halt("Live audit prerequisite failed")
            return False
        try:
            broker.login()
            if not broker.authenticated():
                raise RuntimeError("broker did not expose an authenticated session")
            if self.cfg["mode"] == "live":
                broker_ips = broker.whitelisted_ips()
                configured_ips = set(self.cfg["ip_whitelist"])
                if not broker_ips or not (configured_ips & broker_ips):
                    raise RuntimeError("broker does not confirm a configured static IP")
        except Exception as exc:
            self.halt(f"Broker authentication/compliance check failed: {exc}")
            return False

        key = f"auth_date_{self.cfg['platform']}"
        self.db.kv_set(key, now().strftime("%Y-%m-%d"))
        self.db.ex("INSERT INTO events(kind,msg,ts)VALUES('AUTH','broker session verified',?)", (iso(),))
        self.db.audit("BROKER_SESSION_VERIFIED", {"platform": self.cfg["platform"], "mode": self.cfg["mode"]})
        LOG.info("Broker session and static-IP configuration verified")
        return True

    def wire_kill_switch(self, oms) -> None:
        self.oms = oms

    def halt(self, reason: str) -> None:
        if self.halted:
            return
        self.halted = True
        self.halt_reason = reason
        LOG.critical("COMPLIANCE HALT: %s", reason)
        self.db.ex("INSERT INTO events(kind,msg,ts)VALUES('HALT',?,?)", (reason, iso()))
        self.db.audit("COMPLIANCE_HALT", {"reason": reason[:200]})
        self.db.kv_set("agent_health", {"state": "HALTED", "detail": reason[:160], "ts": iso()})
        if self.oms:
            self.oms.kill_switch(reason)
