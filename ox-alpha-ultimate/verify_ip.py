"""Verify static IP 13.207.244.242 against ox-alpha-ultimate + broker."""
import os, sys, pathlib, ipaddress, tempfile
ip="13.207.244.242"
assert ipaddress.ip_address(ip).is_global, "not global"
print(f"[1] IP {ip} is valid global IPv4")
base=pathlib.Path(__file__).parent
sys.path.insert(0, str(base))
os.environ["DHAN_STATIC_IP"]=ip
from ox.core import Cfg
cfg=Cfg(str(base/"config.yaml"))
wl=cfg["ip_whitelist"]
print(f"[2] Cfg ip_whitelist resolved from DHAN_STATIC_IP: {wl}")
assert ip in wl, "IP not in resolved whitelist - check ip_whitelist_env"
from ox.brokers import make_broker
from ox.core import DB
tmp=pathlib.Path(tempfile.mktemp(suffix=".db"))
db=DB(str(tmp))
broker=make_broker(cfg, db)
print(f"[3] Broker active: {broker.name} (live Dhan will call GET /ip/getIP to confirm)")
if broker.name=="paper":
    print("    (paper mode - broker whitelisted_ips() returns None by design; live Dhan will verify)")
try:
    import requests
    egress=requests.get("https://api.ipify.org", timeout=5).text.strip()
    print(f"[4] Current internet egress IP: {egress}")
    if egress==ip: print("    MATCH - ready for Dhan registration")
    else: print("    MISMATCH - you are on a different network. Register the egress IP instead, or route via your static-IP host.")
except Exception as e:
    print(f"[4] egress check skipped: {e}")
print("\nAll local checks passed. Keep DHAN_STATIC_IP in host env, never in config.yaml or git.")
try: tmp.unlink(missing_ok=True)
except: pass
try: db.close()
except: pass
