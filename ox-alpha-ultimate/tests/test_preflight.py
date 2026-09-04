"""Offline, deterministic tests for ox.preflight (the zero-credential check).

Every network touch is injected: ``http`` and ``public_ip`` stubs, plus a
fake git dict, so none of these tests ever reach the internet, read a real
~/.ox_secrets.env, or depend on the working tree's dirtiness.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ox.preflight import (
    FAIL, PASS, SKIP, WARN,
    _effective_whitelist, _parse_secrets, _secmap_format, evaluate,
)

PROJECT = Path(__file__).resolve().parents[1]

HEAD = "d28d8ee3127128c6f91a47311fc481afa97ed318"
_CLEAN_GIT = {
    "head": HEAD, "origin_master": HEAD, "branch": "master",
    "dirty_files": 0, "git_ok": True,
}


def _http_ok(url: str, timeout: tuple) -> tuple[bool, str]:
    return True, "HTTP 200"


def _http_down(url: str, timeout: tuple) -> tuple[bool, str]:
    return False, "ConnectionError"


def _http_blocked_venue(url: str, timeout: tuple) -> tuple[bool, str]:
    # venue gateways fail while the internet sentinel (ipify) stays up
    if "ipify" in url:
        return True, "HTTP 200"
    return False, "ConnectionError"


def _ip(ok: bool, value: str):
    return lambda: (ok, value)


def _no_secrets(tmp: Path) -> Path:
    return tmp / "missing.env"


def _write_secrets(tmp: Path, body: str) -> Path:
    path = tmp / "secrets.env"
    path.write_text(body, encoding="utf-8")
    return path


def _venue_rows(checks, venue: str, label: str | None = None):
    rows = [c for c in checks if c["venue"] == venue]
    if label:
        rows = [c for c in rows if c["label"] == label]
    return rows


class PreflightUnitTests(unittest.TestCase):
    def test_parse_secrets_reads_exports_without_executing(self):
        text = (
            "# comment\nexport A='one'\nexport B=\"two words\"\n"
            "export C=plain\nnot_an_export=X\nD=$(dangerous)\n"
        )
        parsed = _parse_secrets(text)
        self.assertEqual(parsed, {"A": "one", "B": "two words", "C": "plain"})

    def test_secmap_format_rules_per_venue(self):
        # Dhan: bare numeric securityId only.
        self.assertIsNone(_secmap_format("dhan", {"TCS": "11536"}))
        self.assertIn("numeric", _secmap_format("dhan", {"TCS": "NSE|11536|TCS-EQ"}))
        # Choice: EXCH|TOKEN|TRADINGSYMBOL.
        self.assertIsNone(_secmap_format("choice", {"TCS": "NSE|11536|TCS-EQ"}))
        self.assertIn("EXCH|TOKEN|TRADINGSYMBOL",
                      _secmap_format("choice", {"TCS": "11536"}))
        self.assertIn("EXCH|TOKEN|TRADINGSYMBOL",
                      _secmap_format("choice", {"TCS": "NSE|abc|TCS-EQ"}))

    def test_effective_whitelist_merges_the_ip_env(self):
        cfg = {"ip_whitelist": ["13.207.244.242"],
               "ip_whitelist_env": "DHAN_STATIC_IP"}
        merged = _effective_whitelist(
            cfg, {"DHAN_STATIC_IP": "203.0.113.9,203.0.113.10"})
        # string sort: '203.0.113.10' < '203.0.113.9'
        self.assertEqual(sorted(merged),
                         ["13.207.244.242", "203.0.113.10", "203.0.113.9"])


class PreflightFreshCloneTests(unittest.TestCase):
    """No secrets file anywhere: the check must stay READY (no FAIL) with the
    shipped configs, warning about the setup-live.sh prompts instead."""

    def run_check(self, http=_http_ok, ip=None, secrets_body: str | None = None):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict("os.environ", {}, clear=True):
            tmp = Path(directory)
            secrets = _write_secrets(tmp, secrets_body) if secrets_body is not None \
                else _no_secrets(tmp)
            return evaluate(PROJECT, secrets, http, public_ip=ip or _ip(True, "203.0.113.7"),
                            git=_CLEAN_GIT)

    def test_fresh_clone_is_ready_and_prompts_for_keys(self):
        checks = self.run_check()
        fails = [c for c in checks if c["status"] == FAIL]
        self.assertEqual([], fails, [c["detail"] for c in fails])
        for venue in ("dhan", "choice", "binance"):
            verdict = _venue_rows(checks, venue, "verdict")
            self.assertEqual(len(verdict), 1)
            self.assertIn("setup-live.sh", verdict[0]["detail"])
            # security_map of the shipped configs is well-formed
            secmap = _venue_rows(checks, venue, "security_map")
            if venue != "binance":
                self.assertTrue(secmap, f"{venue} had no security_map check")
                self.assertEqual(secmap[0]["status"], PASS)
        # Choice's shipped config runs without a depth feed.
        flow = _venue_rows(checks, "choice", "order_flow")
        self.assertEqual(flow[0]["status"], PASS)

    def test_choice_primary_order_flow_fails(self):
        """A choice config that leaves order_flow.primary on is a hard FAIL:
        with no depth feed every entry would be blocked."""
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict("os.environ", {}, clear=True):
            tmp = Path(directory)
            # Only config_choice.yaml exists here; other venues report their
            # missing config as FAIL - we assert only the order-flow rule.
            (tmp / "config_choice.yaml").write_text(
                "mode: paper\nplatform: paper\norder_flow:\n  enabled: true\n"
                "  primary: true\n", encoding="utf-8")
            checks = evaluate(tmp, _no_secrets(tmp), _http_ok,
                              public_ip=_ip(True, "203.0.113.7"), git=_CLEAN_GIT)
        flow = _venue_rows(checks, "choice", "order_flow")
        self.assertEqual(flow[0]["status"], FAIL)
        self.assertIn("primary: false", flow[0]["detail"])

    def test_security_map_drift_fails(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict("os.environ", {}, clear=True):
            tmp = Path(directory)
            (tmp / "config.yaml").write_text(
                "mode: paper\nplatform: paper\nsymbols: [RELIANCE, TCS]\n"
                "security_map:\n  RELIANCE: '2885'\n", encoding="utf-8")
            checks = evaluate(tmp, _no_secrets(tmp), _http_ok,
                              public_ip=_ip(True, "203.0.113.7"), git=_CLEAN_GIT)
        drift = [c for c in checks if c["venue"] == "dhan"
                 and c["label"] == "security_map"]
        self.assertEqual(drift[0]["status"], FAIL)
        self.assertIn("TCS", drift[0]["detail"])


class PreflightSecretsMismatchTests(unittest.TestCase):
    """A secrets file that exists but contradicts the machine/venue must FAIL
    with actionable detail - that is exactly what the live boot would halt on."""

    SECRETS = (
        "export DHAN_CLIENT_ID='c'\nexport DHAN_TOKEN='t'\n"
        "export DHAN_STATIC_IP='198.51.100.9'\n"
        "export BINANCE_API_KEY='k'\n"
        "export CHOICE_USER_ID='u'\nexport CHOICE_PASSWORD='p'\n"
        "export CHOICE_TOTP='123456'\nexport CHOICE_VENDOR_CODE='vc'\n"
        "export CHOICE_API_KEY='ak'\n"
    )

    def test_egress_mismatch_and_missing_key_fail(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict("os.environ", {}, clear=True):
            tmp = Path(directory)
            secrets = _write_secrets(tmp, self.SECRETS)
            checks = evaluate(PROJECT, secrets, _http_ok,
                              public_ip=_ip(True, "203.0.113.7"), git=_CLEAN_GIT)
        # Machine egress != DHAN_STATIC_IP in the secrets file: boot halts.
        for venue in ("dhan", "choice"):
            egress = _venue_rows(checks, venue, "egress IP")
            self.assertEqual(egress[0]["status"], FAIL, egress[0]["detail"])
            self.assertIn("203.0.113.7", egress[0]["detail"])
        # Binance secret is missing from an existing secrets file.
        keys = _venue_rows(checks, "binance", "keys")
        self.assertEqual(keys[0]["status"], FAIL)
        self.assertIn("BINANCE_API_SECRET", keys[0]["detail"])


class PreflightOfflineTests(unittest.TestCase):
    """No internet must degrade to SKIP, never FAIL: preflight stays useful
    and deterministic on an offline machine."""

    def test_offline_skips_egress_and_gateway(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict("os.environ", {}, clear=True):
            tmp = Path(directory)
            checks = evaluate(PROJECT, _no_secrets(tmp), _http_down,
                              public_ip=_ip(False, "ConnectionError"),
                              git=_CLEAN_GIT)
        fails = [c for c in checks if c["status"] == FAIL]
        self.assertEqual([], fails, [c["detail"] for c in fails])
        for venue in ("dhan", "choice", "binance"):
            gateway = _venue_rows(checks, venue, "gateway")
            self.assertEqual(gateway[0]["status"], SKIP)
        # The egress gate exists only where a live platform runs (dhan/choice;
        # promax has no legacy-style egress posture in its own config).
        for venue in ("dhan", "choice"):
            egress = _venue_rows(checks, venue, "egress IP")
            self.assertEqual(egress[0]["status"], SKIP)

    def test_blocked_venue_with_internet_up_is_a_failure(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict("os.environ", {}, clear=True):
            tmp = Path(directory)
            checks = evaluate(PROJECT, _no_secrets(tmp), _http_blocked_venue,
                              public_ip=_ip(True, "203.0.113.7"),
                              git=_CLEAN_GIT)
        for venue in ("dhan", "choice", "binance"):
            gateway = _venue_rows(checks, venue, "gateway")
            self.assertEqual(gateway[0]["status"], FAIL, gateway[0]["detail"])
            self.assertIn("internet is up", gateway[0]["detail"])

    def test_git_drift_is_reported_as_warn_not_fail(self):
        drifted = dict(_CLEAN_GIT)
        drifted["head"] = "0" * 40
        drifted["dirty_files"] = 3
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict("os.environ", {}, clear=True):
            tmp = Path(directory)
            checks = evaluate(PROJECT, _no_secrets(tmp), _http_ok,
                              public_ip=_ip(True, "203.0.113.7"), git=drifted)
        head = [c for c in checks if c["label"] == "git HEAD"][0]
        tree = [c for c in checks if c["label"] == "git working tree"][0]
        self.assertEqual(head["status"], WARN)
        self.assertEqual(tree["status"], WARN)


if __name__ == "__main__":
    unittest.main()
