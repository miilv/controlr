"""ping a sam_masks-protocol side server: ping_side.py SOCK -> exit 0 when it answers the info op"""
import sys
from multiprocessing.connection import Client
c = Client(sys.argv[1], family="AF_UNIX")
c.send({"op": "info"})
r = c.recv()
ok = isinstance(r, dict) and "error" not in r
if len(sys.argv) > 2:
    ok = ok and sorted(r.get("models", [])) == sorted(sys.argv[2].split(","))
c.close()
sys.exit(0 if ok else 1)
