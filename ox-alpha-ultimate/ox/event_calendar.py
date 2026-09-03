"""Earnings/Event Avoidance Calendar."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from enum import Enum
from datetime import datetime, timedelta, date
import json
import requests
from .core import iso, LOG


class EventType(Enum):
    EARNINGS = "earnings"
    DIVIDEND = "dividend"
    SPLIT = "split"
    BONUS = "bonus"
    BOARD_MEETING = "board_meeting"
    AGM = "agm"
    ECONOMIC = "economic"
    HOLIDAY = "holiday"
    EXPIRY = "expiry"
    CUSTOM = "custom"


class EventImpact(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CalendarEvent:
    """Calendar event."""
    event_id: str
    symbol: str
    event_type: EventType
    impact: EventImpact
    event_date: date
    event_time: Optional[str] = None
    details: Dict = field(default_factory=dict)
    source: str = "manual"
    created_at: str = field(default_factory=lambda: iso())


@dataclass
class AvoidanceRule:
    """Rule for avoiding positions around events."""
    event_type: EventType
    impact: EventImpact
    pre_bars: int  # Bars before event to avoid
    post_bars: int  # Bars after event to avoid
    action: str  # "reduce", "close", "block_new"
    max_position_pct: float = 0.0  # Max position as % of portfolio


class EventCalendar:
    """Manages event calendar and position avoidance."""
    
    def __init__(self, cfg, db):
        self.cfg = cfg
        self.db = db
        self.cal_cfg = cfg.get("event_calendar", {})
        self.enabled = self.cal_cfg.get("enabled", True)
        self.auto_fetch = self.cal_cfg.get("auto_fetch", True)
        self.fetch_interval_hours = self.cal_cfg.get("fetch_interval_hours", 24)
        
        # Default avoidance rules
        self.rules = {
            (EventType.EARNINGS, EventImpact.HIGH): AvoidanceRule(
                EventType.EARNINGS, EventImpact.HIGH, 5, 2, "close", 0.0
            ),
            (EventType.EARNINGS, EventImpact.MEDIUM): AvoidanceRule(
                EventType.EARNINGS, EventImpact.MEDIUM, 3, 1, "reduce", 0.5
            ),
            (EventType.EARNINGS, EventImpact.LOW): AvoidanceRule(
                EventType.EARNINGS, EventImpact.LOW, 2, 1, "block_new", 1.0
            ),
            (EventType.DIVIDEND, EventImpact.MEDIUM): AvoidanceRule(
                EventType.DIVIDEND, EventImpact.MEDIUM, 2, 1, "reduce", 0.75
            ),
            (EventType.EXPIRY, EventImpact.HIGH): AvoidanceRule(
                EventType.EXPIRY, EventImpact.HIGH, 3, 0, "close", 0.0
            ),
            (EventType.ECONOMIC, EventImpact.CRITICAL): AvoidanceRule(
                EventType.ECONOMIC, EventImpact.CRITICAL, 2, 2, "block_new", 0.5
            ),
        }
        
        # Override with config
        for rule_cfg in self.cal_cfg.get("avoidance_rules", []):
            key = (EventType(rule_cfg["event_type"]), EventImpact(rule_cfg["impact"]))
            self.rules[key] = AvoidanceRule(
                key[0], key[1],
                rule_cfg.get("pre_bars", 2),
                rule_cfg.get("post_bars", 1),
                rule_cfg.get("action", "block_new"),
                rule_cfg.get("max_position_pct", 0.5)
            )
        
        self._events: Dict[str, List[CalendarEvent]] = defaultdict(list)
        self._last_fetch = 0.0
    
    def add_event(self, event: CalendarEvent):
        """Add event to calendar."""
        self._events[event.symbol].append(event)
        self._events[event.symbol].sort(key=lambda e: e.event_date)
        
        # Persist to database
        self.db.ex("""
            INSERT OR REPLACE INTO calendar_events
            (event_id, symbol, event_type, impact, event_date, event_time, details, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.event_id,
            event.symbol,
            event.event_type.value,
            event.impact.value,
            event.event_date.isoformat(),
            event.event_time,
            json.dumps(event.details),
            event.source,
            event.created_at
        ))
    
    def load_events(self):
        """Load events from database."""
        rows = self.db.q("""
            SELECT event_id, symbol, event_type, impact, event_date, event_time, details, source, created_at
            FROM calendar_events
            WHERE event_date >= date('now', '-30 days')
        """)
        
        for row in rows:
            event = CalendarEvent(
                event_id=row[0],
                symbol=row[1],
                event_type=EventType(row[2]),
                impact=EventImpact(row[3]),
                event_date=date.fromisoformat(row[4]),
                event_time=row[5],
                details=json.loads(row[6]) if row[6] else {},
                source=row[7],
                created_at=row[8]
            )
            self._events[event.symbol].append(event)
    
    def get_events_for_symbol(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[CalendarEvent]:
        """Get events for a symbol within date range."""
        events = self._events.get(symbol, [])
        
        if start_date:
            events = [e for e in events if e.event_date >= start_date]
        if end_date:
            events = [e for e in events if e.event_date <= end_date]
        
        return events
    
    def get_upcoming_events(self, days: int = 7) -> Dict[str, List[CalendarEvent]]:
        """Get all upcoming events within days."""
        today = date.today()
        end_date = today + timedelta(days=days)
        
        result = {}
        for symbol, events in self._events.items():
            upcoming = [e for e in events if today <= e.event_date <= end_date]
            if upcoming:
                result[symbol] = upcoming
        
        return result
    
    def check_avoidance(
        self,
        symbol: str,
        current_date: date,
        current_position_pct: float = 0.0
    ) -> Tuple[bool, str, Optional[AvoidanceRule]]:
        """Check if position should be avoided/reduced for symbol."""
        if not self.enabled:
            return False, "calendar_disabled", None
        
        events = self.get_events_for_symbol(symbol, current_date, current_date + timedelta(days=30))
        
        for event in events:
            rule = self.rules.get((event.event_type, event.impact))
            if not rule:
                continue
            
            # Check pre-event window
            pre_start = event.event_date - timedelta(days=rule.pre_bars)
            post_end = event.event_date + timedelta(days=rule.post_bars)
            
            if pre_start <= current_date <= post_end:
                action_msg = f"{rule.action}_{event.event_type.value}_{event.impact.value}"
                
                if rule.action == "close":
                    return True, action_msg, rule
                elif rule.action == "reduce" and current_position_pct > rule.max_position_pct:
                    return True, action_msg, rule
                elif rule.action == "block_new":
                    return True, action_msg, rule
        
        return False, "ok", None
    
    def get_position_adjustment(
        self,
        symbol: str,
        current_date: date,
        current_position_pct: float
    ) -> Tuple[float, str]:
        """Get recommended position adjustment (0-1 multiplier)."""
        should_avoid, reason, rule = self.check_avoidance(symbol, current_date, current_position_pct)
        
        if not should_avoid:
            return 1.0, "ok"
        
        if rule.action == "close":
            return 0.0, reason
        elif rule.action == "reduce":
            return rule.max_position_pct, reason
        elif rule.action == "block_new":
            return 1.0 if current_position_pct > 0 else 0.0, reason
        
        return 1.0, reason
    
    def fetch_earnings_calendar(self, symbols: List[str]) -> int:
        """Fetch earnings calendar from external source."""
        if not self.auto_fetch:
            return 0
        
        added = 0
        # This would integrate with NSE/BSE APIs or financial data providers
        # For now, return 0
        return added
    
    def fetch_economic_calendar(self) -> int:
        """Fetch economic calendar (RBI policy, CPI, GDP, etc.)."""
        # Would integrate with economic calendar APIs
        return 0
    
    def add_custom_event(
        self,
        symbol: str,
        event_type: EventType,
        impact: EventImpact,
        event_date: date,
        details: Optional[Dict] = None
    ) -> CalendarEvent:
        """Add custom event."""
        event = CalendarEvent(
            event_id=f"custom_{symbol}_{event_date.isoformat()}_{event_type.value}",
            symbol=symbol,
            event_type=event_type,
            impact=impact,
            event_date=event_date,
            details=details or {},
            source="manual"
        )
        self.add_event(event)
        return event
    
    def remove_event(self, event_id: str) -> bool:
        """Remove event from calendar."""
        for symbol, events in self._events.items():
            for i, event in enumerate(events):
                if event.event_id == event_id:
                    events.pop(i)
                    self.db.ex("DELETE FROM calendar_events WHERE event_id=?", (event_id,))
                    return True
        return False
    
    def get_avoidance_schedule(self, days: int = 30) -> List[Dict]:
        """Get schedule of avoidance periods."""
        today = date.today()
        end_date = today + timedelta(days=days)
        
        schedule = []
        for symbol, events in self._events.items():
            for event in events:
                if today <= event.event_date <= end_date:
                    rule = self.rules.get((event.event_type, event.impact))
                    if rule:
                        schedule.append({
                            "symbol": symbol,
                            "event_type": event.event_type.value,
                            "impact": event.impact.value,
                            "event_date": event.event_date.isoformat(),
                            "avoidance_start": (event.event_date - timedelta(days=rule.pre_bars)).isoformat(),
                            "avoidance_end": (event.event_date + timedelta(days=rule.post_bars)).isoformat(),
                            "action": rule.action,
                            "max_position_pct": rule.max_position_pct
                        })
        
        return sorted(schedule, key=lambda x: x["avoidance_start"])


class EconomicCalendar:
    """Indian economic calendar with RBI policy, CPI, GDP, IIP, etc."""
    
    KEY_EVENTS = [
        {"name": "RBI Monetary Policy", "type": EventType.ECONOMIC, "impact": EventImpact.CRITICAL},
        {"name": "CPI Inflation", "type": EventType.ECONOMIC, "impact": EventImpact.HIGH},
        {"name": "WPI Inflation", "type": EventType.ECONOMIC, "impact": EventImpact.MEDIUM},
        {"name": "GDP Release", "type": EventType.ECONOMIC, "impact": EventImpact.HIGH},
        {"name": "IIP Data", "type": EventType.ECONOMIC, "impact": EventImpact.MEDIUM},
        {"name": "PMI Manufacturing", "type": EventType.ECONOMIC, "impact": EventImpact.MEDIUM},
        {"name": "PMI Services", "type": EventType.ECONOMIC, "impact": EventImpact.MEDIUM},
        {"name": "Trade Balance", "type": EventType.ECONOMIC, "impact": EventImpact.LOW},
        {"name": "Fiscal Deficit", "type": EventType.ECONOMIC, "impact": EventImpact.MEDIUM},
    ]
    
    def __init__(self, calendar: EventCalendar):
        self.calendar = calendar
    
    def populate_known_events(self, year: int = None):
        """Populate known recurring economic events."""
        year = year or date.today().year
        
        # RBI policy dates (typically bi-monthly)
        rbi_months = [2, 4, 6, 8, 10, 12]
        for month in rbi_months:
            # First week of month, typically Wednesday/Thursday
            for day in range(1, 8):
                try:
                    event_date = date(year, month, day)
                    if event_date.weekday() in [2, 3]:  # Wed/Thu
                        self.calendar.add_custom_event(
                            symbol="MARKET",
                            event_type=EventType.ECONOMIC,
                            impact=EventImpact.CRITICAL,
                            event_date=event_date,
                            details={"name": "RBI Monetary Policy", "recurring": True}
                        )
                        break
                except ValueError:
                    continue
        
        # Monthly data releases (approximate)
        monthly_events = {
            "CPI Inflation": (12, EventImpact.HIGH),  # ~12th of month
            "WPI Inflation": (14, EventImpact.MEDIUM),  # ~14th
            "IIP Data": (12, EventImpact.MEDIUM),  # ~12th
        }
        
        for name, (day, impact) in monthly_events.items():
            for month in range(1, 13):
                try:
                    event_date = date(year, month, day)
                    self.calendar.add_custom_event(
                        symbol="MARKET",
                        event_type=EventType.ECONOMIC,
                        impact=impact,
                        event_date=event_date,
                        details={"name": name, "recurring": True}
                    )
                except ValueError:
                    continue
        
        # Quarterly GDP (approximate)
        gdp_months = [2, 5, 8, 11]  # Month after quarter end
        for month in gdp_months:
            for day in range(25, 31):
                try:
                    event_date = date(year, month, day)
                    if event_date.weekday() < 5:  # Weekday
                        self.calendar.add_custom_event(
                            symbol="MARKET",
                            event_type=EventType.ECONOMIC,
                            impact=EventImpact.HIGH,
                            event_date=event_date,
                            details={"name": "GDP Release", "recurring": True}
                        )
                        break
                except ValueError:
                    continue


class ExpiryCalendar:
    """F&O Expiry calendar."""
    
    def __init__(self, calendar: EventCalendar):
        self.calendar = calendar
    
    def populate_expiries(self, year: int = None, months: int = 12):
        """Populate monthly and weekly expiries."""
        year = year or date.today().year
        
        # Monthly expiry: Last Thursday of each month
        for month in range(1, 13):
            # Find last Thursday
            last_day = 31
            while True:
                try:
                    test_date = date(year, month, last_day)
                    break
                except ValueError:
                    last_day -= 1
            
            # Go backwards to find Thursday
            for day in range(last_day, max(1, last_day-6), -1):
                try:
                    event_date = date(year, month, day)
                    if event_date.weekday() == 3:  # Thursday
                        self.calendar.add_custom_event(
                            symbol="NIFTY",
                            event_type=EventType.EXPIRY,
                            impact=EventImpact.HIGH,
                            event_date=event_date,
                            details={"name": "Monthly Expiry", "index": "NIFTY"}
                        )
                        self.calendar.add_custom_event(
                            symbol="BANKNIFTY",
                            event_type=EventType.EXPIRY,
                            impact=EventImpact.HIGH,
                            event_date=event_date,
                            details={"name": "Monthly Expiry", "index": "BANKNIFTY"}
                        )
                        break
                except ValueError:
                    continue
        
        # Weekly expiries (every Thursday for Nifty, Wednesday for BankNifty)
        # This would be too many events, so only add near-term
        today = date.today()
        for weeks in range(8):  # Next 8 weeks
            for offset in range(7):
                check_date = today + timedelta(days=weeks*7 + offset)
                if check_date.weekday() == 3:  # Thursday
                    self.calendar.add_custom_event(
                        symbol="NIFTY",
                        event_type=EventType.EXPIRY,
                        impact=EventImpact.MEDIUM,
                        event_date=check_date,
                        details={"name": "Weekly Expiry", "index": "NIFTY"}
                    )
                elif check_date.weekday() == 2:  # Wednesday
                    self.calendar.add_custom_event(
                        symbol="BANKNIFTY",
                        event_type=EventType.EXPIRY,
                        impact=EventImpact.MEDIUM,
                        event_date=check_date,
                        details={"name": "Weekly Expiry", "index": "BANKNIFTY"}
                    )