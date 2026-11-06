"""ping a sam_masks-protocol side server: ping_side.py SOCK -> exit 0 when it answers the info op"""
import sys
from multiprocessing.connection import Client
c = Client(sys.argv[1], family="AF_UNIX")
c.send({"op": "info"})
r = c.recv()
sys.exit(0 if isinstance(r, dict) else 1)
