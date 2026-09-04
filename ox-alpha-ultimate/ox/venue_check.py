"""Login-only venue rehearsal for the live boundary.

``check_venue`` proves a venue's REAL authentication handshake (network,
credentials, gateway) without placing, modifying, or cancelling any order.
It is the last offline-provable bridge between the scripted-transport suite
and the first supervised live session: run it from the deployment instance
via ``python run.py venue-check`` (or ``bash scripts/live.sh verify-all``)
once credentials exist.  It never invents a fill and never touches a position.
"""

from __future__ import annotations

from typing import Callable

from .brokers import BrokerBase


def check_venue(name: str, builder: Callable[[], BrokerBase]) -> tuple[str, str]:
    """Rehearse one venue: returns (status, detail) with status in PASS/FAIL.

    A venue is SKIPped by the caller when its credentials are absent; this
    function only ever reports PASS or FAIL for a venue it is asked to build.
    """
    try:
        broker = builder()
        broker.login()
        if not broker.authenticated():
            return "FAIL", "login returned but no authenticated session"
        return "PASS", f"{broker.name} session established"
    except Exception as exc:  # noqa: BLE001 - surface any gate failure verbatim
        return "FAIL", f"{exc.__class__.__name__}: {exc}"