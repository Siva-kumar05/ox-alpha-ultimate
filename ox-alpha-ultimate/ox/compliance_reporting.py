"""Automated Compliance Reporting."""
from __future__ import annotations
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from .core import LOG, iso, DB


class ReportType(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    ON_DEMAND = "on_demand"


class ReportFormat(Enum):
    JSON = "json"
    PDF = "pdf"
    HTML = "html"
    CSV = "csv"


@dataclass
class ComplianceReport:
    """Compliance report."""
    report_id: str
    report_type: ReportType
    period_start: str
    period_end: str
    generated_at: str
    format: ReportFormat
    file_path: str
    sections: Dict
    status: str = "generated"


class ComplianceReporter:
    """Generates automated compliance reports."""
    
    def __init__(self, cfg, db: DB):
        self.cfg = cfg
        self.db = db
        self.report_cfg = cfg.get("compliance_reporting", {})
        self.enabled = self.report_cfg.get("enabled", True)
        reports_dir = Path(self.report_cfg.get("reports_dir", "compliance_reports"))
        if not reports_dir.is_absolute():
            # Anchor under the database directory (the agent's state dir) so
            # reports never leak into the process CWD, e.g. during test runs.
            try:
                base = Path(self.db.path).expanduser().resolve().parent
            except AttributeError:
                base = Path.cwd()
            reports_dir = base / reports_dir
        self.reports_dir = reports_dir
        self.default_format = ReportFormat(self.report_cfg.get("default_format", "json"))
        self.schedule = self.report_cfg.get("schedule", {
            "daily": "06:00",
            "weekly": "mon 06:00",
            "monthly": "1 06:00"
        })
        
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._scheduled_reports: Dict[str, threading.Thread] = {}
        self._lock = threading.RLock()
    
    def generate_report(
        self,
        report_type: ReportType,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
        format: Optional[ReportFormat] = None
    ) -> ComplianceReport:
        """Generate a compliance report."""
        format = format or self.default_format
        
        # Determine period
        now = datetime.now()
        if period_end is None:
            period_end = now.isoformat()
        if period_start is None:
            if report_type == ReportType.DAILY:
                period_start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0).isoformat()
            elif report_type == ReportType.WEEKLY:
                period_start = (now - timedelta(weeks=1)).replace(hour=0, minute=0, second=0).isoformat()
            elif report_type == ReportType.MONTHLY:
                period_start = (now - timedelta(days=30)).replace(hour=0, minute=0, second=0).isoformat()
            else:
                period_start = (now - timedelta(days=1)).isoformat()
        
        # Collect data
        sections = self._collect_report_data(period_start, period_end)
        
        # Create report
        report_id = f"{report_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        file_name = f"{report_id}.{format.value}"
        file_path = self.reports_dir / file_name
        
        # Generate content
        content = {
            "report_id": report_id,
            "report_type": report_type.value,
            "period_start": period_start,
            "period_end": period_end,
            "generated_at": iso(),
            "format": format.value,
            "sections": sections
        }
        
        # Write file
        if format == ReportFormat.JSON:
            with open(file_path, "w") as f:
                json.dump(content, f, indent=2, default=str)
        elif format == ReportFormat.CSV:
            self._write_csv(content, file_path)
        
        report = ComplianceReport(
            report_id=report_id,
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            generated_at=iso(),
            format=format,
            file_path=str(file_path),
            sections=sections
        )
        
        LOG.info(f"Compliance report generated: {file_path}")
        return report

    # ── cadence driver (called at agent boot) ─────────────────────────────
    # The tick loop is not running at the configured 06:00 report times, so
    # the agent evaluates due reports at boot.  Completion is tracked per
    # report type with a kv marker holding the last generated period date, so
    # each report is generated at most once per period even across restarts.
    _DONE_PREFIX = "compliance_done_"
    _WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}

    def _marker(self, report_type: ReportType) -> str:
        value = self.db.kv_get(self._DONE_PREFIX + report_type.value)
        return str(value) if value is not None else ""

    @staticmethod
    def _schedule_token(schedule: dict, key: str, default: str) -> str:
        raw = str(schedule.get(key, default)).strip().split()
        return raw[0].lower() if raw else default.strip().split()[0].lower()

    def _due(self, report_type: ReportType) -> bool:
        schedule = self.schedule if isinstance(self.schedule, dict) else {}
        today = datetime.now()
        marker = self._marker(report_type)
        if report_type == ReportType.DAILY:
            # Covers the previous day; due until a report for it exists.
            return marker != (today - timedelta(days=1)).strftime("%Y-%m-%d")
        if report_type == ReportType.WEEKLY:
            token = self._schedule_token(schedule, "weekly", "mon 06:00")
            if self._WEEKDAYS.get(token, 0) != today.weekday():
                return False
            if marker:
                try:
                    return datetime.strptime(marker, "%Y-%m-%d").date() < (today - timedelta(days=6)).date()
                except ValueError:
                    return True
            return True
        if report_type == ReportType.MONTHLY:
            token = self._schedule_token(schedule, "monthly", "1 06:00")
            try:
                day_of_month = int(token)
            except ValueError:
                day_of_month = 1
            if today.day != day_of_month:
                return False
            if marker:
                try:
                    return datetime.strptime(marker, "%Y-%m-%d").date() < (today - timedelta(days=27)).date()
                except ValueError:
                    return True
            return True
        return False

    def run_due_reports(self) -> List[str]:
        """Generate every report whose configured cadence is due at this boot.

        Fail-closed: each report is attempted independently; an error is
        logged and that period simply stays due for the next boot.  Returns
        the paths of the reports generated in this call.
        """
        if not self.enabled:
            return []
        generated: List[str] = []
        today_iso = datetime.now().strftime("%Y-%m-%d")
        for report_type in (ReportType.DAILY, ReportType.WEEKLY, ReportType.MONTHLY):
            try:
                if not self._due(report_type):
                    continue
                report = self.generate_report(report_type)
                # The marker records the period the report covers, so a daily
                # report generated at Monday's boot marks Sunday and stays
                # skipped for the rest of Monday even across restarts.
                if report_type == ReportType.DAILY:
                    marker_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                else:
                    marker_date = today_iso
                self.db.kv_set(self._DONE_PREFIX + report_type.value, marker_date)
                generated.append(str(report.file_path))
            except Exception as exc:  # noqa: BLE001 - one report must not break the cadence
                LOG.error("Compliance %s report generation failed: %s", report_type.value, exc)
        return generated
    
    def _collect_report_data(self, period_start: str, period_end: str) -> Dict:
        """Collect all data for compliance report."""
        sections = {}
        
        # Trade summary
        sections["trade_summary"] = self._get_trade_summary(period_start, period_end)
        
        # P&L summary
        sections["pnl_summary"] = self._get_pnl_summary(period_start, period_end)
        
        # Risk metrics
        sections["risk_metrics"] = self._get_risk_metrics(period_start, period_end)
        
        # Strategy performance
        sections["strategy_performance"] = self._get_strategy_performance(period_start, period_end)
        
        # Order flow metrics
        sections["order_flow_metrics"] = self._get_order_flow_metrics(period_start, period_end)
        
        # Audit trail
        sections["audit_trail"] = self._get_audit_trail(period_start, period_end)
        
        # Compliance checks
        sections["compliance_checks"] = self._get_compliance_checks(period_start, period_end)
        
        # Incidents
        sections["incidents"] = self._get_incidents(period_start, period_end)
        
        return sections
    
    def _get_trade_summary(self, start: str, end: str) -> Dict:
        """Get trade summary."""
        rows = self.db.q("""
            SELECT COUNT(*), SUM(qty), SUM(pnl), SUM(charges)
            FROM trades WHERE intime >= ? AND intime <= ?
        """, (start, end))
        
        if rows:
            count, total_qty, total_pnl, total_charges = rows[0]
            return {
                "total_trades": count or 0,
                "total_quantity": total_qty or 0,
                "total_pnl": float(total_pnl or 0),
                "total_charges": float(total_charges or 0),
                "net_pnl": float((total_pnl or 0) - (total_charges or 0))
            }
        return {}
    
    def _get_pnl_summary(self, start: str, end: str) -> Dict:
        """Get P&L summary by day."""
        rows = self.db.q("""
            SELECT date(intime) as trade_date, COUNT(*) as trades, SUM(pnl) as pnl, SUM(charges) as charges
            FROM trades WHERE intime >= ? AND intime <= ?
            GROUP BY date(intime) ORDER BY trade_date
        """, (start, end))
        
        daily = []
        for date_str, trades, pnl, charges in rows:
            daily.append({
                "date": date_str,
                "trades": trades,
                "gross_pnl": float(pnl or 0),
                "charges": float(charges or 0),
                "net_pnl": float((pnl or 0) - (charges or 0))
            })
        
        return {"daily": daily}
    
    def _get_risk_metrics(self, start: str, end: str) -> Dict:
        """Get risk metrics."""
        # Get equity curve
        equity_rows = self.db.q("""
            SELECT ts, equity FROM equity WHERE ts >= ? AND ts <= ? ORDER BY ts
        """, (start, end))
        
        if equity_rows:
            equities = [float(r[1]) for r in equity_rows]
            returns = []
            for i in range(1, len(equities)):
                if equities[i-1] > 0:
                    returns.append((equities[i] - equities[i-1]) / equities[i-1])
            
            if returns:
                import numpy as np
                returns_arr = np.array(returns)
                return {
                    "sharpe": float(np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252)) if np.std(returns_arr) > 0 else 0,
                    "max_drawdown": float(np.min((np.array(equities) - np.maximum.accumulate(equities)) / np.maximum.accumulate(equities))),
                    "volatility": float(np.std(returns_arr) * np.sqrt(252)),
                    "var_95": float(-np.percentile(returns_arr, 5)) if len(returns_arr) > 20 else 0
                }
        
        return {}
    
    def _get_strategy_performance(self, start: str, end: str) -> Dict:
        """Get strategy performance."""
        rows = self.db.q("""
            SELECT strat, COUNT(*) as trades, SUM(pnl) as pnl, AVG(pnl) as avg_pnl
            FROM trades WHERE intime >= ? AND intime <= ?
            GROUP BY strat
        """, (start, end))
        
        strategies = {}
        for strat, trades, pnl, avg_pnl in rows:
            strategies[strat] = {
                "trades": trades,
                "total_pnl": float(pnl or 0),
                "avg_pnl": float(avg_pnl or 0)
            }
        
        return strategies
    
    def _get_order_flow_metrics(self, start: str, end: str) -> Dict:
        """Get order flow metrics."""
        rows = self.db.q("""
            SELECT AVG(spread_bps), AVG(book_imbalance), AVG(flow_imbalance), 
                   AVG(pressure_ema), AVG(liquidity_score)
            FROM orderflow WHERE ts >= ? AND ts <= ?
        """, (start, end))
        
        if rows and rows[0][0] is not None:
            return {
                "avg_spread_bps": float(rows[0][0]),
                "avg_book_imbalance": float(rows[0][1]),
                "avg_flow_imbalance": float(rows[0][2]),
                "avg_pressure_ema": float(rows[0][3]),
                "avg_liquidity_score": float(rows[0][4])
            }
        return {}
    
    def _get_audit_trail(self, start: str, end: str) -> List[Dict]:
        """Get audit trail."""
        rows = self.db.q("""
            SELECT action, payload, ts FROM audit WHERE ts >= ? AND ts <= ? ORDER BY ts
        """, (start, end))
        
        return [
            {"action": r[0], "payload": r[1], "timestamp": r[2]}
            for r in rows
        ]
    
    def _get_compliance_checks(self, start: str, end: str) -> Dict:
        """Get compliance check results."""
        # IP checks
        ip_rows = self.db.q("""
            SELECT msg, ts FROM events WHERE kind='IP_CHECK' AND ts >= ? AND ts <= ?
        """, (start, end))
        
        ip_checks = [{"message": r[0], "timestamp": r[1]} for r in ip_rows]
        
        # Auth checks
        auth_rows = self.db.q("""
            SELECT msg, ts FROM events WHERE kind='AUTH' AND ts >= ? AND ts <= ?
        """, (start, end))
        
        auth_checks = [{"message": r[0], "timestamp": r[1]} for r in auth_rows]
        
        # Halts
        halt_rows = self.db.q("""
            SELECT msg, ts FROM events WHERE kind='HALT' AND ts >= ? AND ts <= ?
        """, (start, end))
        
        halts = [{"message": r[0], "timestamp": r[1]} for r in halt_rows]
        
        return {
            "ip_checks": ip_checks,
            "auth_checks": auth_checks,
            "halts": halts
        }
    
    def _get_incidents(self, start: str, end: str) -> List[Dict]:
        """Get incidents."""
        rows = self.db.q("""
            SELECT kind, msg, ts FROM events WHERE kind IN ('KILL', 'FAILOVER', 'ALPHA_DECAY_ALERT') AND ts >= ? AND ts <= ?
        """, (start, end))
        
        return [{"type": r[0], "message": r[1], "timestamp": r[2]} for r in rows]
    
    def _write_csv(self, content: Dict, file_path: Path):
        """Write report as CSV."""
        import csv
        
        # Flatten sections for CSV
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Section", "Key", "Value"])
            
            for section_name, section_data in content.get("sections", {}).items():
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        writer.writerow([section_name, key, str(value)])
                elif isinstance(section_data, list):
                    for i, item in enumerate(section_data):
                        if isinstance(item, dict):
                            for key, value in item.items():
                                writer.writerow([section_name, f"{i}.{key}", str(value)])
                        else:
                            writer.writerow([section_name, str(i), str(item)])
    
    def schedule_reports(self):
        """Schedule automatic reports."""
        # This would use a scheduler like APScheduler
        # For now, just log
        LOG.info("Report scheduling configured (would use APScheduler)")
    
    def list_reports(self) -> List[Dict]:
        """List generated reports."""
        reports = []
        for file_path in self.reports_dir.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    reports.append({
                        "file": file_path.name,
                        "report_id": data.get("report_id"),
                        "type": data.get("report_type"),
                        "period_start": data.get("period_start"),
                        "period_end": data.get("period_end"),
                        "generated_at": data.get("generated_at")
                    })
            except Exception:
                pass
        return sorted(reports, key=lambda r: r.get("generated_at", ""), reverse=True)