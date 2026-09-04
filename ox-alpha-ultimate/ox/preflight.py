"""Zero-credential readiness check for a live run - ``python run.py preflight``.

Everything that can be checked WITHOUT credentials is checked here, so the
user runs it BEFORE entering any keys.  Credentials themselves are never
read from the chat or the command line: they belong in ~/.ox_secrets.env,
written by ``scripts/setup-live.sh`` with hidden input.  This module only
reports which of those keys setup-live.sh will still prompt for.

Network access is best-effort and offline-safe: gateway and egress probes
that cannot complete are reported SKIP (never counted as a failure), so the
check is deterministic on a machine without internet.

Exit code: 0 when no check FAILed, 1 otherwise (WARN/SKIP never fail the
run).  Verdicts are printed per venue: dhan, choice, binance.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from importlib import util as _import_util
from pathlib import Path
from typing import Callable

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"

PUBLIC_IP_URL = "https://api.ipify.org"
INTERNET_SENTINEL = PUBLIC_IP_URL  # same host doubles as the offline sentinel

HTTP = Callable[[str, tuple], tuple[bool, str]]
"""http(url, timeout) -> (ok, detail).  Injected so tests never touch the net."""

# name -> (config file, live platform, gateway host)
_VENUE_SPECS = {
    "dhan": ("config.yaml", "dhan", "https://api.dhan.co"),
    "choice": ("config_choice.yaml", "choice", "https://api.shoonya.com"),
    "binance": ("config_promax.yaml", None, "https://api.binance.com"),
}

# Keys each venue's live command needs (setup-live.sh prompts for these).
_VENUE_KEYS = {
    "dhan": ["DHAN_CLIENT_ID", "DHAN_TOKEN"],
    "choice": ["CHOICE_USER_ID", "CHOICE_PASSWORD", "CHOICE_TOTP",
               "CHOICE_VENDOR_CODE", "CHOICE_API_KEY"],
    "binance": ["BINANCE_API_KEY", "BINANCE_API_SECRET"],
}

# Core dependencies; a missing one means `pip install -r requirements.txt`.
_CORE_DEPS = ["numpy", "pandas", "yaml", "requests"]
_OPTIONAL_DEPS = {"ccxt": "binance live crypto (requirements.txt ships it)"}


def _default_http(url: str, timeout: tuple) -> tuple[bool, str]:
    try:
        import requests

        response = requests.get(url, timeout=timeout, allow_redirects=True)
        return True, f"HTTP {response.status_code}"
    except requests.exceptions.Timeout as exc:
        return False, f"timeout {exc}"
    except Exception as exc:  # noqa: BLE001 - any transport failure is a probe result
        return False, exc.__class__.__name__


def _default_public_ip() -> tuple[bool, str]:
    """(ok, public-ip-or-error).  Injectable so tests never touch the net."""
    try:
        import requests

        ip = requests.get(PUBLIC_IP_URL, timeout=(3.05, 5.0)).text.strip()
        import ipaddress

        ipaddress.ip_address(ip)
        return True, ip
    except Exception as exc:  # noqa: BLE001
        return False, exc.__class__.__name__


def _parse_secrets(text: str) -> dict[str, str]:
    """Parse KEY='value' export lines WITHOUT executing the file."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("export "):
            continue
        try:
            key, _, value = line[len("export "):].partition("=")
        except ValueError:  # pragma: no cover - defensive
            continue
        if key and value:
            try:
                values[key] = shlex.split(value)[0] if shlex.split(value) else ""
            except ValueError:
                values[key] = value.strip("'\"")
    return values


def _read_secrets(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        return _parse_secrets(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return {}


def _git_info(root: Path) -> dict:
    """Local git facts used for drift detection (never touches the network)."""
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    head = run("rev-parse", "HEAD")
    origin = run("rev-parse", "origin/master")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    dirty = run("status", "--porcelain")
    return {
        "head": head or None,
        "origin_master": origin or None,
        "branch": branch or None,
        "dirty_files": len([line for line in dirty.splitlines() if line]),
        "git_ok": bool(head),
    }


def _one(venue: str, label: str, status: str, detail: str) -> dict:
    return {"venue": venue, "label": label, "status": status, "detail": detail}


def _env_union(secrets: dict[str, str]) -> dict[str, str]:
    merged = dict(os.environ)
    for key, value in secrets.items():
        if value and key not in merged:
            merged[key] = value
    return merged


def _secmap_format(spec: str, mapping: dict) -> str | None:
    """None when every entry matches the venue's format; else an error text."""
    for sym, entry in mapping.items():
        value = str(entry).strip()
        if spec == "dhan":
            if not value.isdigit():
                return f"{sym}: {entry!r} is not a numeric Dhan securityId"
        else:  # choice: EXCH|TOKEN|TRADINGSYMBOL
            parts = value.split("|")
            if len(parts) != 3 or not parts[0] or not parts[1].isdigit() or not parts[2]:
                return f"{sym}: {entry!r} is not 'EXCH|TOKEN|TRADINGSYMBOL' (e.g. NSE|2885|RELIANCE-EQ)"
    return None


def _effective_whitelist(cfg: dict, env: dict) -> list[str]:
    listed = [str(ip).strip() for ip in cfg.get("ip_whitelist", []) or []]
    ip_env_name = str(cfg.get("ip_whitelist_env", "") or "")
    if ip_env_name:
        listed.extend(
            part.strip()
            for part in env.get(ip_env_name, "").split(",")
            if part.strip()
        )
    return listed


def evaluate(root: Path, secrets_file: Path, http: HTTP,
             public_ip: Callable[[], tuple[bool, str]] | None = None,
             git: dict | None = None) -> list[dict]:
    """Run every no-credential check; returns flat check records."""
    checks: list[dict] = []
    secrets = _read_secrets(secrets_file)
    env = _env_union(secrets)
    git = git if git is not None else _git_info(root)
    public_ip = public_ip if public_ip is not None else _default_public_ip
    have_secrets = bool(secrets)

    # -- environment ------------------------------------------------------ #
    missing_deps = [
        name for name in _CORE_DEPS
        if _import_util.find_spec(name) is None
    ]
    if missing_deps:
        checks.append(_one("environment", "dependencies", FAIL,
                           f"missing: {', '.join(missing_deps)} - run "
                           "'pip install -r requirements.txt'"))
    else:
        checks.append(_one("environment", "dependencies", PASS,
                           "core deps importable (numpy/pandas/yaml/requests)"))
    for name, purpose in _OPTIONAL_DEPS.items():
        if _import_util.find_spec(name) is None:
            checks.append(_one("environment", f"optional dep {name}", WARN,
                               f"not installed - {purpose}"))
    if getattr(sys, "base_prefix", None) and sys.prefix == sys.base_prefix:
        checks.append(_one("environment", "python venv", WARN,
                           "not running inside a venv (activate .venv first)"))
    else:
        checks.append(_one("environment", "python venv", PASS, "venv active"))

    root_lower = str(root).lower()
    if any(token in root_lower for token in ("onedrive", "dropbox")):
        checks.append(_one("environment", "project location", WARN,
                           f"live run from a cloud-synced path ({root}); "
                           "file locks can corrupt the SQLite state - clone to "
                           "a plain path such as C:\\ox-alpha-src"))
    else:
        checks.append(_one("environment", "project location", PASS,
                           "not under a cloud-synced folder"))

    if git.get("git_ok"):
        if git.get("origin_master"):
            if git.get("head") == git.get("origin_master"):
                checks.append(_one("environment", "git HEAD", PASS,
                                   f"{git['head'][:7]} matches origin/master"))
            else:
                checks.append(_one("environment", "git HEAD", WARN,
                                   f"local HEAD {str(git.get('head'))[:7]} != "
                                   f"origin/master {str(git.get('origin_master'))[:7]} "
                                   "- run 'git pull' or clone fresh"))
        else:
            checks.append(_one("environment", "git HEAD", SKIP,
                               "no origin/master ref locally; drift not checked"))
        dirty = int(git.get("dirty_files") or 0)
        if dirty:
            checks.append(_one("environment", "git working tree", WARN,
                               f"{dirty} uncommitted file(s) - clone fresh for "
                               "the clean deployed tree"))
        else:
            checks.append(_one("environment", "git working tree", PASS,
                               "clean"))
    else:
        checks.append(_one("environment", "git", SKIP, "not a git checkout"))

    if not secrets:
        checks.append(_one("environment", "credentials", WARN,
                           "no ~/.ox_secrets.env yet - bash scripts/setup-live.sh "
                           "will prompt (hidden input) for every venue key below"))
    # -- per venue --------------------------------------------------------- #
    for venue, (config_name, live_platform, gateway) in _VENUE_SPECS.items():
        config_path = root / config_name
        checks.append(_one(venue, "config file", PASS if config_path.exists() else FAIL,
                           config_name if config_path.exists() else f"{config_name} missing"))
        cfg: dict = {}
        if config_path.exists():
            try:
                import yaml

                cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                checks.append(_one(venue, "yaml", PASS, "parses"))
            except Exception as exc:  # noqa: BLE001
                checks.append(_one(venue, "yaml", FAIL,
                                   f"{exc.__class__.__name__}: {exc}"))
                continue

        mode = str(cfg.get("mode", ""))
        platform = str(cfg.get("platform", ""))
        if live_platform is not None:
            if mode not in ("paper", "live"):
                checks.append(_one(venue, "config mode/platform", FAIL,
                                   f"mode={mode!r} must be 'paper' or 'live'"))
            if platform not in ("paper", live_platform):
                checks.append(_one(venue, "config mode/platform", FAIL,
                                   f"platform={platform!r}; {venue} runs "
                                   f"platform: {live_platform} (live.sh flips it)"))
            else:
                checks.append(_one(venue, "config mode/platform", PASS,
                                   f"platform {platform} valid for {venue}"))

        mapping = cfg.get("security_map")
        symbols = {str(s).upper() for s in cfg.get("symbols", []) or []}
        if live_platform is not None and isinstance(mapping, dict):
            normalized = {str(k).upper() for k in mapping}
            missing_sym = sorted(symbols - normalized)
            extra = sorted(k for k in normalized if k not in symbols)
            problem = _secmap_format("dhan" if venue == "dhan" else "choice", mapping)
            if problem:
                checks.append(_one(venue, "security_map", FAIL, problem))
            elif missing_sym or extra:
                checks.append(_one(venue, "security_map", FAIL,
                                   f"symbols/security_map drift - missing: "
                                   f"{', '.join(missing_sym) or '-'}, extra: "
                                   f"{', '.join(extra) or '-'}"))
            else:
                checks.append(_one(venue, "security_map", PASS,
                                   f"{len(mapping)} entries mirror the {len(symbols)} "
                                   "configured symbols"))
        elif live_platform is not None:
            checks.append(_one(venue, "security_map", WARN,
                               "no security_map - live equity quotes will fail "
                               "closed on every lookup"))

        if venue == "choice":
            flow = cfg.get("order_flow", {}) or {}
            if flow.get("primary") is True:
                checks.append(_one(venue, "order_flow", FAIL,
                                   "order_flow.primary: true - Choice has no depth "
                                   "feed; every entry would be blocked "
                                   "(ORDER_FLOW_UNAVAILABLE). Set primary: false."))
            else:
                checks.append(_one(venue, "order_flow", PASS,
                                   "primary not true (Choice runs on LTP + candles)"))
        if venue == "binance":
            crypto = cfg.get("crypto", {}) or {}
            if not crypto.get("markets"):
                checks.append(_one(venue, "crypto markets", WARN,
                                   "crypto.markets empty - no crypto symbols defined"))
            else:
                checks.append(_one(venue, "crypto markets", PASS,
                                   f"{len(crypto['markets'])} market(s) configured"))
            if not cfg.get("security_map"):
                checks.append(_one(venue, "equity security_map", WARN,
                                   "absent - promax's Dhan equity side cannot resolve a "
                                   "live quote (crypto agents are unaffected); add Dhan "
                                   "securityIds from your portal, never guess them"))
            else:
                checks.append(_one(venue, "equity security_map", PASS,
                                   f"{len(cfg['security_map'])} Dhan securityId(s) configured"))

        # -- live posture (what boot would reject) ------------------------ #
        if live_platform is not None and mode == "paper":
            whitelist = _effective_whitelist(cfg, env)
            expected = env.get("DHAN_STATIC_IP", "").strip()
            ok, ip_or_error = public_ip()
            if not ok:
                checks.append(_one(venue, "egress IP", SKIP,
                                   f"public IP lookup failed ({ip_or_error}) - "
                                   "egress not verified (offline-safe)"))
            elif ip_or_error in whitelist:
                checks.append(_one(venue, "egress IP", PASS,
                                   f"{ip_or_error} is whitelisted for the live boot"))
            elif expected and ip_or_error != expected:
                checks.append(_one(venue, "egress IP", FAIL,
                                   f"egress is {ip_or_error} but DHAN_STATIC_IP is "
                                   f"{expected}; the live boot egress check will halt"))
            elif expected and not whitelist:
                checks.append(_one(venue, "egress IP", FAIL,
                                   f"DHAN_STATIC_IP={expected} is set but the config's "
                                   "ip_whitelist_env does not merge it (check the config)"))
            elif not expected:
                checks.append(_one(venue, "egress IP",
                                   FAIL if have_secrets else WARN,
                                   "no DHAN_STATIC_IP set - export your public IP "
                                   "(https://api.ipify.org) or the live boot halts"))
            else:
                checks.append(_one(venue, "egress IP", WARN,
                                   f"egress {ip_or_error} not in allowlist {whitelist}; "
                                   "export DHAN_STATIC_IP with the current IP"))
            if not whitelist and not expected:
                checks.append(_one(venue, "live whitelist", WARN,
                                   "live boot needs one whitelisted IP "
                                   "(DHAN_STATIC_IP env + ip_whitelist_env in config)"))

        # -- gateway reachability (offline-safe) -------------------------- #
        ok, detail = http(gateway, (3.05, 4.0))
        if ok:
            checks.append(_one(venue, "gateway", PASS, f"{gateway} reachable ({detail})"))
        else:
            sentinel_ok, _ = http(INTERNET_SENTINEL, (3.05, 4.0))
            if sentinel_ok:
                checks.append(_one(venue, "gateway", FAIL,
                                   f"{gateway} unreachable ({detail}) while the "
                                   "internet is up"))
            else:
                checks.append(_one(venue, "gateway", SKIP,
                                   f"{gateway} not probed - no internet detected "
                                   "(offline or blocked)"))

        # -- keys (never read from chat; reported as setup-live.sh prompts) -#
        required = _VENUE_KEYS[venue]
        present = [key for key in required if env.get(key, "").strip()]
        missing = [key for key in required if not env.get(key, "").strip()]
        if venue in ("binance", "dhan"):
            # promax always logs in the Dhan equity side; legacy Dhan is self-evident.
            if venue == "binance":
                dhan_missing = [key for key in _VENUE_KEYS["dhan"]
                                if not env.get(key, "").strip()]
                if dhan_missing:
                    checks.append(_one(venue, "keys", FAIL if have_secrets else WARN,
                                       "promax also logs in Dhan equity at boot - "
                                       f"missing: {', '.join(dhan_missing)}"))
        if missing:
            if have_secrets:
                checks.append(_one(venue, "keys", FAIL,
                                   f"missing in ~/.ox_secrets.env: {', '.join(missing)} "
                                   "- re-run bash scripts/setup-live.sh"))
            else:
                checks.append(_one(venue, "keys", WARN,
                                   "setup-live.sh will prompt for: " +
                                   ", ".join(required)))
        else:
            checks.append(_one(venue, "keys", PASS,
                               "all present: " + ", ".join(required)))

    # -- verdicts ---------------------------------------------------------- #
    for venue in _VENUE_SPECS:
        rows = [c for c in checks if c["venue"] == venue]
        fails = [c for c in rows if c["status"] == FAIL]
        warns = [c for c in rows if c["status"] == WARN]
        if fails:
            verdict = f"NOT READY - {len(fails)} FAIL, {len(warns)} WARN"
        elif not have_secrets:
            verdict = "READY - run bash scripts/setup-live.sh to enter keys"
        else:
            verdict = f"READY ({len(warns)} WARN)"
        checks.append(_one(venue, "verdict", FAIL if fails else PASS, verdict))
    return checks


def main(argv: list[str] | None = None) -> int:
    """Print the report; returns the process exit code."""
    root = Path(__file__).resolve().parents[1]
    secrets_file = Path(os.getenv("OX_SECRETS_FILE", "~/.ox_secrets.env")).expanduser()
    checks = evaluate(root, secrets_file, _default_http, _default_public_ip)

    print("OX-ALPHA PREFLIGHT - zero-credential readiness (keys are never read "
          "from chat or the command line)")
    print("=" * 72)
    for check in checks:
        print(f"[{check['status']:<4}] {check['venue']:<12} {check['label']:<24} "
              f"{check['detail']}")
    fails = [c for c in checks if c["status"] == FAIL]
    skips = [c for c in checks if c["status"] == SKIP]
    print("=" * 72)
    print(f"PREFLIGHT: {len(fails)} FAIL, {len(skips)} SKIP (SKIP = offline, not counted)")
    if not fails:
        print("Next: bash scripts/setup-live.sh   (hidden-input prompts; stores "
              "~/.ox_secrets.env, chmod 600)")
        print("      bash scripts/live.sh verify-all   (login-only, no orders)")
    print("Keys belong ONLY in ~/.ox_secrets.env via setup-live.sh - never paste "
          "API keys into chat, email, or the command line.")
    return 1 if fails else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
