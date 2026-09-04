# OX-ALPHA Operational Runbooks

## Table of Contents
1. [System Overview](#system-overview)
2. [Daily Operations](#daily-operations)
3. [Emergency Procedures](#emergency-procedures)
4. [Troubleshooting Guide](#troubleshooting-guide)
5. [Maintenance Procedures](#maintenance-procedures)
6. [Monitoring & Alerting](#monitoring--alerting)
7. [Backup & Recovery](#backup--recovery)
8. [Configuration Management](#configuration-management)
9. [Security Procedures](#security-procedures)
10. [Incident Response](#incident-response)

---

## System Overview

### Architecture Components
- **Agent Core** (`ox/agent.py`): Main trading orchestrator
- **Brain** (`ox/brain.py`): Strategy generation and validation
- **Risk Manager** (`ox/risk.py`): Portfolio risk and position sizing
- **OMS** (`ox/oms.py`): Order management with broker-side brackets
- **Brokers** (`ox/brokers.py`): Dhan (live), Paper (testing)
- **Order Flow** (`ox/orderflow.py`): L2 depth analysis
- **Regime Detector** (`ox/regime.py`): Market regime classification
- **Stop Manager** (`ox/stop_manager.py`): Advanced stop types
- **Position Sizer** (`ox/position_sizing.py`): Kelly, correlation-aware sizing
- **Health/Metrics** (`ox/health_metrics.py`): HTTP endpoints
- **Failover** (`ox/failover.py`): Multi-broker failover
- **Shutdown** (`ox/graceful_shutdown.py`): Graceful shutdown handling

### Key Data Stores
- SQLite database (`oxalpha.db`): Trades, positions, strategies, candles, audit
- Configuration (`config.yaml`): All system parameters
- Logs (`oxalpha.log`): Structured JSON logs

---

## Daily Operations

### Morning Startup (Pre-Market)
```bash
# 1. Verify system health
curl http://localhost:8080/health

# 2. Check broker connectivity
curl http://localhost:8080/health/ready

# 3. Review overnight events
sqlite3 oxalpha.db "SELECT * FROM events WHERE ts >= date('now','start of day');"

# 4. Verify strategies loaded
sqlite3 oxalpha.db "SELECT sid, template, status, score FROM strategies WHERE status='LIVE_APPROVED';"

# 5. Check capital and risk limits
sqlite3 oxalpha.db "SELECT * FROM equity ORDER BY ts DESC LIMIT 1;"
```

### Market Hours Monitoring
```bash
# Real-time health (every 30s)
watch -n 30 'curl -s http://localhost:8080/health | jq .'

# Position monitoring
sqlite3 oxalpha.db "SELECT sym, qty, avg, sl, tp, strat FROM positions;"

# P&L tracking
sqlite3 oxalpha.db "SELECT SUM(pnl) as daily_pnl FROM trades WHERE intime >= date('now','start of day');"
```

### End of Day
```bash
# 1. Verify EOD squareoff
sqlite3 oxalpha.db "SELECT * FROM events WHERE kind='EOD_SQUAREOFF' AND ts >= date('now','start of day');"

# 2. Generate daily compliance report
python -c "from ox.compliance_reporting import ComplianceReporter; from ox.core import Cfg, DB; cfg=Cfg(); db=DB(cfg['db_path']); r=ComplianceReporter(cfg,db).generate_report(ReportType.DAILY)"

# 3. Backup database
python -c "from ox.database_backup import DatabaseBackupManager; from ox.core import Cfg; cfg=Cfg(); b=DatabaseBackupManager(cfg, cfg['db_path']); b.create_backup()"

# 4. Review alpha decay alerts
sqlite3 oxalpha.db "SELECT * FROM events WHERE kind='ALPHA_DECAY_ALERT' AND ts >= date('now','start of day');"
```

---

## Live Launch (Dhan / Binance)

Credentials are stored **once** in `~/.ox_secrets.env` (chmod 600) by an
interactive prompt; re-run it any time a key changes:

```bash
bash scripts/setup-live.sh        # prompts: Dhan client ID/token, instance IP, Binance key/secret
```

Launch with one command - the launcher sources the secrets file, flips the
right config to `mode: live`, and runs the venue:

```bash
bash scripts/live.sh live-test    # Dhan connectivity + credential check (exit 0/2, safe)
bash scripts/live.sh dhan         # legacy NSE intraday agent, live Dhan
bash scripts/live.sh binance      # promax orchestrator: Dhan equity + Binance spot/perp
bash scripts/live.sh track        # track record
bash scripts/live.sh status       # positions / strategies / recent trades
bash scripts/live.sh paper        # paper boot (reverts the config edits)
```

Gates that must be true before the first live order:

- `OX_LIVE_EXECUTION_APPROVED` is written by the setup script (never type
  real money through a session that lacks it).
- The Dhan IP whitelist (`ip_whitelist` + `DHAN_STATIC_IP` env) contains the
  instance's **public** IP; the boot's egress check verifies it.
- `OX_AUDIT_KEY` is auto-generated when left blank.
- Binance live additionally needs `OX_PROMAX_AUTO_APPROVE` **unset** so every
  order goes through `python run.py intents` / `ok <iid>` human approval.

### Choice India (Shoonya) live

`platform: choice` runs the legacy NSE intraday agent through the real
**ChoiceBroker** adapter (Finvasia Shoonya/Noren gateway). Transport and
payload contract are mirrored from the official Shoonya wrapper
(`Shoonya-Dev/ShoonyaApi-py`) and its `NorenRestApiPy` base: form POSTs of
`jData=<json>&jKey=<token>` to `https://api.shoonya.com/NorenWClientTP/`;
login exchanges SHA-256 hashed password + `uid|apikey` app key for a session
token. Credentials (prompted by `setup-live.sh`):

- `CHOICE_USER_ID` / `CHOICE_PASSWORD` / `CHOICE_TOTP` (TPIN)
- `CHOICE_VENDOR_CODE` / `CHOICE_API_KEY` (from your broker)
- `CHOICE_IMEI` (defaults to `ox-alpha-ultimate`)

**Before launching**, rewrite `security_map` entries to the Shoonya form
`EXCH|TOKEN|TRADINGSYMBOL` (the Dhan numeric ids will fail closed):

```yaml
security_map:
  RELIANCE: NSE|2885|RELIANCE-EQ
  TCS:      NSE|11536|TCS-EQ
```

Then `bash scripts/live.sh choice` (or set `mode: live` + `platform: choice`
manually). Known adapter limits (each fails closed, never silent):

- **No depth/order-flow feed** - decisions run on LTP + candles; keep
  `order_flow.primary: false` or every entry is blocked as
  `ORDER_FLOW_UNAVAILABLE`.
- **No Dhan-style Super Order leg modification** - breakeven target
  adjustments raise; the OMS falls back to its enforced-target logic and
  targets are protected by the broker-side stop (`place_protective_stop`).
- **No IP whitelist** - Shoonya authenticates by session token, so the
  Dhan static-IP confirmation is skipped (the egress `check_ip` gate still
  applies if `ip_whitelist` is configured).
- **Live-credential verification is unproven** - everything here is tested
  offline against a scripted transport; the first live `login()` against the
  real gateway (and the exact order-book/position field names) still needs a
  supervised run with real Shoonya credentials. Until then, treat Choice as
  demo-ready, not money-ready.

---

## Emergency Procedures

### Kill Switch Activation
```bash
# Automatic: KILL.flag created by system
# Manual: Create KILL.flag
echo "MANUAL KILL: $(date)" > KILL.flag

# Verify kill switch executed
sqlite3 oxalpha.db "SELECT * FROM events WHERE kind='KILL' ORDER BY ts DESC LIMIT 5;"
```

### Broker Failover
```bash
# Automatic failover is configured
# Manual failover to paper:
curl -X POST http://localhost:8080/admin/failover/paper

# Check current broker
curl http://localhost:8080/health | jq .brokers.current_broker

# Force failback to primary when recovered
curl -X POST http://localhost:8080/admin/failover/dhan
```

### Circuit Breaker Halt
```bash
# Check circuit breaker state
curl http://localhost:8080/health | jq '.checks[] | select(.name=="circuit_breaker")'

# If HALT state, investigate:
# 1. Check recent P&L
sqlite3 oxalpha.db "SELECT SUM(pnl) FROM trades WHERE intime >= date('now','-7 days');"
# 2. Check Sharpe degradation
sqlite3 oxalpha.db "SELECT msg FROM events WHERE kind='ALPHA_DECAY_ALERT' ORDER BY ts DESC LIMIT 10;"
# 3. Manual override (if justified):
#    Edit config.yaml: set self_healing.l3_sharpe_threshold lower
#    Restart agent
```

### Data Feed Failure
```bash
# Check order flow status
curl http://localhost:8080/health | jq '.checks[] | select(.name=="data_freshness")'

# If stale data:
# 1. Check broker connectivity
# 2. Restart order flow feed
# 3. Check websocket connection logs
grep "Dhan depth feed" oxalpha.log | tail -20
```

---

## Troubleshooting Guide

### Common Issues

#### 1. "No validated strategies loaded"
**Symptoms**: Agent starts but `validated_strategies=0`
**Causes**:
- No strategies passed walk-forward validation
- Strategies quarantined due to schema mismatch
- Training hasn't run yet

**Resolution**:
```bash
# Check strategy status
sqlite3 oxalpha.db "SELECT sid, status, score FROM strategies;"

# Run training manually
python -c "from ox.core import Cfg, DB; from ox.agent import Agent; cfg=Cfg(); db=DB(cfg['db_path']); a=Agent(); a.nightly_training()"

# Check training data sufficiency
sqlite3 oxalpha.db "SELECT sym, COUNT(*) FROM candles GROUP BY sym;"
```

#### 2. "Order flow unavailable"
**Symptoms**: Entries blocked with `ORDER_FLOW_UNAVAILABLE`
**Causes**:
- Websocket disconnected
- Depth feed not started
- Stale depth data

**Resolution**:
```bash
# Check depth feed status
curl http://localhost:8080/health | jq '.checks[] | select(.name=="broker_connection")'

# Check websocket logs
grep "Dhan depth feed" oxalpha.log | tail -10

# Restart order flow
python -c "from ox.core import Cfg, DB; from ox.brokers import DhanBroker; cfg=Cfg(); db=DB(cfg['db_path']); b=DhanBroker(cfg,db); b.login(); b.start_orderflow()"
```

#### 3. "Risk gate rejected"
**Symptoms**: Entries blocked with `RISK_GATE`
**Causes**:
- Daily loss cap reached
- Max positions reached
- Portfolio VaR exceeded
- Loss streak cooldown

**Resolution**:
```bash
# Check risk metrics
sqlite3 oxalpha.db "SELECT * FROM kv WHERE k='portfolio_stats';"

# Check daily P&L
sqlite3 oxalpha.db "SELECT SUM(pnl) FROM trades WHERE intime >= date('now','start of day');"

# Check positions
sqlite3 oxalpha.db "SELECT COUNT(*) FROM positions;"

# Check loss streak
sqlite3 oxalpha.db "SELECT pnl FROM trades ORDER BY tid DESC LIMIT 10;"
```

#### 4. "Database locked"
**Symptoms**: SQLite `database is locked` errors
**Causes**:
- Multiple processes accessing DB
- Long-running transaction
- WAL mode issues

**Resolution**:
```bash
# Check for other processes
lsof oxalpha.db

# Kill other processes
pkill -f "python.*agent"

# Check WAL files
ls -la oxalpha.db*

# Restart agent cleanly
./start-daily.cmd
```

#### 5. "Broker authentication failed"
**Symptoms**: `AuthenticationError` on startup
**Causes**:
- Expired token
- Invalid credentials
- IP not whitelisted

**Resolution**:
```bash
# Check token validity
curl -H "Authorization: Bearer $DHAN_TOKEN" https://api.dhan.co/v2/fundlimit

# Verify IP whitelist
curl https://api.ipify.org

# Regenerate token (if using TOTP)
python -c "import pyotp; print(pyotp.TOTP('$DHAN_TOTP_SECRET').now())"
```

---

## Maintenance Procedures

### Weekly Maintenance
```bash
# 1. Database vacuum
sqlite3 oxalpha.db "VACUUM;"

# 2. Clean old candles (keep 120 days)
sqlite3 oxalpha.db "DELETE FROM candles WHERE ts < strftime('%s', date('now','-120 days'));"

# 3. Clean old orderflow (keep 30 days)
sqlite3 oxalpha.db "DELETE FROM orderflow WHERE ts < date('now','-30 days');"

# 4. Verify backup integrity
python -c "from ox.database_backup import DatabaseBackupManager; from ox.core import Cfg; cfg=Cfg(); b=DatabaseBackupManager(cfg, cfg['db_path']); print(b.get_status())"

# 5. Review strategy performance
sqlite3 oxalpha.db "SELECT strat, COUNT(*), AVG(pnl), SUM(pnl) FROM trades WHERE intime >= date('now','-7 days') GROUP BY strat;"
```

### Monthly Maintenance
```bash
# 1. Rotate secrets (if not auto)
python -c "from ox.secret_rotation import SecretManager; from ox.core import Cfg; cfg=Cfg(); m=SecretManager(cfg); m.rotate_secret('dhan_token')"

# 2. Retrain strategies
python -c "from ox.core import Cfg, DB; from ox.agent import Agent; cfg=Cfg(); db=DB(cfg['db_path']); a=Agent(); a.nightly_training()"

# 3. Archive old logs
mv oxalpha.log oxalpha.log.$(date +%Y%m)
gzip oxalpha.log.$(date +%Y%m)

# 4. Update dependencies
pip install --upgrade -r requirements.txt

# 5. Run full benchmark
python -c "from ox.load_testing import create_benchmark_runner; from ox.agent import Agent; a=Agent(); b=create_benchmark_runner(a); b.run_full_benchmark()"
```

### Quarterly Maintenance
```bash
# 1. Full system audit
# 2. Review and update risk parameters
# 3. Validate compliance reports
# 4. Chaos engineering experiments
# 5. Disaster recovery drill
```

---

## Monitoring & Alerting

### Health Check Endpoints
| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `GET /health` | Full health check | 200 OK with all checks |
| `GET /health/ready` | Readiness probe | 200 if ready, 503 if not |
| `GET /health/live` | Liveness probe | 200 always |
| `GET /metrics` | JSON metrics | 200 with metrics |
| `GET /metrics/prometheus` | Prometheus format | 200 with metrics |

### Key Metrics to Monitor
| Metric | Warning Threshold | Critical Threshold |
|--------|------------------|-------------------|
| `rolling_sharpe` | < 0.5 | < 0 |
| `daily_loss_pct` | > 1% | > 2% |
| `position_count` | > 3 | > 5 |
| `broker_latency_ms` | > 500ms | > 2000ms |
| `data_staleness_sec` | > 5s | > 10s |
| `circuit_breaker_state` | L1_REDUCE_SIZE | L2_STOP_ENTRIES |

### Alert Channels
- **Critical**: PagerDuty / SMS
- **Warning**: Slack / Email
- **Info**: Log aggregation

---

## Backup & Recovery

### Backup Schedule
- **Every 6 hours**: Automated database backup (compressed)
- **Daily**: Configuration backup
- **Weekly**: Full system backup (DB + config + logs)

### Recovery Procedures

#### Database Restore
```bash
# 1. Stop agent
./stop-agent.sh

# 2. List available backups
python -c "from ox.database_backup import DatabaseBackupManager; from ox.core import Cfg; cfg=Cfg(); b=DatabaseBackupManager(cfg, cfg['db_path']); [print(b.backup_id, b.timestamp) for b in b.list_backups()]"

# 3. Restore specific backup
python -c "from ox.database_backup import DatabaseBackupManager; from ox.core import Cfg; cfg=Cfg(); b=DatabaseBackupManager(cfg, cfg['db_path']); b.restore_backup('backup_20260115_060000')"

# 4. Verify restore
sqlite3 oxalpha.db "SELECT COUNT(*) FROM trades;"

# 5. Restart agent
./start-daily.cmd
```

#### Configuration Restore
```bash
# Restore config from git
git checkout HEAD -- config.yaml

# Or from backup
cp config.yaml.backup.$(date +%Y%m%d) config.yaml

# Hot-reload will pick up changes automatically
```

---

## Configuration Management

### Hot-Reloadable Parameters
| Parameter | Section | Reload Type |
|-----------|---------|-------------|
| `risk.risk_per_trade_pct` | risk | Immediate |
| `execution.trailing_jump` | execution | Immediate |
| `order_flow.min_book_imbalance` | order_flow | Immediate |
| `stop_management.default_atr_mult` | stop_management | Next position |
| `position_sizing.order_flow.max_multiplier` | position_sizing | Next calculation |

### Non-Reloadable Parameters (Require Restart)
| Parameter | Section |
|-----------|---------|
| `symbols` | root |
| `capital` | root |
| `mode` | root |
| `platform` | root |
| `security_map` | root |
| `ip_whitelist` | root |

### Configuration Validation
```bash
# Validate config
python -c "from ox.core import Cfg; Cfg('config.yaml'); print('Config valid')"

# Check for required fields
python -c "
from ox.core import Cfg
cfg = Cfg('config.yaml')
required = ['symbols', 'capital', 'risk.risk_per_trade_pct', 'risk.max_positions']
for r in required:
    keys = r.split('.')
    val = cfg.d
    for k in keys:
        val = val.get(k)
    print(f'{r}: {val}')
"
```

---

## Security Procedures

### Secret Rotation
```bash
# Check secret status
python -c "from ox.secret_rotation import create_default_secret_manager; from ox.core import Cfg; cfg=Cfg(); m=create_default_secret_manager(cfg); print(m.get_status())"

# Manual rotation
python -c "from ox.secret_rotation import create_default_secret_manager; from ox.core import Cfg; cfg=Cfg(); m=create_default_secret_manager(cfg); m.rotate_secret('dhan_token')"

# Verify new secret works
python -c "from ox.brokers import DhanBroker; from ox.core import Cfg, DB; cfg=Cfg(); db=DB(cfg['db_path']); b=DhanBroker(cfg,db); print(b.login())"
```

### IP Whitelist Management
```bash
# Current IPs
sqlite3 oxalpha.db "SELECT msg FROM events WHERE kind='IP_CHECK' ORDER BY ts DESC LIMIT 1;"

# Add new IP (update config.yaml)
# config.yaml: ip_whitelist: ["1.2.3.4", "5.6.7.8"]

# Or via environment variable
export DHAN_STATIC_IP="1.2.3.4,5.6.7.8"

# Hot-reload will pick up
```

### Audit Trail Verification
```bash
# Verify audit chain
python -c "from ox.core import Cfg, DB; cfg=Cfg(); db=DB(cfg['db_path']); print('Audit valid:', db.verify_audit())"

# Export audit for compliance
sqlite3 oxalpha.db "SELECT * FROM audit ORDER BY aid;" > audit_export.csv
```

---

## Incident Response

### Severity Levels

| Severity | Definition | Response Time | Escalation |
|----------|------------|---------------|------------|
| SEV-1 | Trading halted, data loss, security breach | 15 min | Page on-call |
| SEV-2 | Degraded performance, partial outage | 1 hour | Notify team |
| SEV-3 | Minor issue, workaround exists | 4 hours | Ticket |
| SEV-4 | Cosmetic, no impact | Next business day | Backlog |

### Incident Response Flow

1. **Detect**: Alert fires or manual report
2. **Triage**: Assess severity, impact
3. **Communicate**: Notify stakeholders
4. **Mitigate**: Apply workaround/fix
5. **Resolve**: Root cause fix
6. **Postmortem**: Document, learn

### Common Incident Scenarios

#### SEV-1: Kill Switch Activated
```bash
# 1. Check KILL.flag content
cat KILL.flag

# 2. Check audit log for reason
sqlite3 oxalpha.db "SELECT * FROM audit WHERE action='KILL_SWITCH' ORDER BY aid DESC LIMIT 1;"

# 3. Check OMS kill switch events
sqlite3 oxalpha.db "SELECT * FROM events WHERE kind='KILL' ORDER BY ts DESC;"

# 4. Assess positions at kill time
sqlite3 oxalpha.db "SELECT * FROM trades WHERE exit_reason='KILL_SWITCH';"

# 5. Only restart after root cause identified
rm KILL.flag
./start-daily.cmd
```

#### SEV-1: Broker API Failure
```bash
# 1. Check failover status
curl http://localhost:8080/health | jq .brokers

# 2. If primary failed, verify failover broker
curl http://localhost:8080/health | jq '.checks[] | select(.name=="broker_connection")'

# 3. Check broker error logs
grep "BrokerError\|OrderError" oxalpha.log | tail -20

# 4. Contact broker support if needed
# 5. Monitor for auto-failback
```

#### SEV-2: Alpha Decay Detected
```bash
# 1. Check affected strategies
sqlite3 oxalpha.db "SELECT * FROM events WHERE kind='ALPHA_DECAY_ALERT' ORDER BY ts DESC LIMIT 5;"

# 2. Review strategy performance
sqlite3 oxalpha.db "SELECT strat, COUNT(*), AVG(pnl) FROM trades WHERE intime >= date('now','-30 days') GROUP BY strat;"

# 3. Consider retiring strategy
python -c "from ox.post_trade_analysis import PostTradeAnalyzer; from ox.core import Cfg, DB; cfg=Cfg(); db=DB(cfg['db_path']); a=PostTradeAnalyzer(cfg,db); print(a.should_retire_strategy('strategy_id'))"

# 4. Quarantine if necessary
sqlite3 oxalpha.db "UPDATE strategies SET status='QUARANTINED' WHERE sid='strategy_id';"
```

---

## Contact Information

| Role | Contact | Escalation |
|------|---------|------------|
| Primary On-Call | [Name/Phone] | 15 min |
| Secondary On-Call | [Name/Phone] | 30 min |
| Team Lead | [Name/Phone] | 1 hour |
| Broker Support (Dhan) | support@dhan.co | As needed |
| Infrastructure | [Name/Phone] | 1 hour |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-15 | Initial runbook |
| 1.1 | 2026-02-01 | Added chaos engineering procedures |
| 1.2 | 2026-03-01 | Added compliance reporting section |

---

*Last Updated: 2026-08-31*
*Next Review: 2026-11-30*