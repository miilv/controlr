"""dojo.runtime: the one policy that wires perception, WorldState, expert programs, twin verification and
execution into a proxy_eval policy (``RuntimePolicy``: ``reset(info)`` / ``next(obs)``).

* ``policy.py``      the episode loop (perceive -> look -> supervisor -> program with verify -> resync -> finish)
* ``supervisor.py``  the skill library and the instruction-only choice (no per-task routing)
* ``belief.py``      the skills provider: GT rows (mode gt, dev only) or a belief (mode perception)
* ``verifier.py``    the skills TwinBackend over the bundle's twin process, loaded from OUR estimate
* ``bridge.py``      synchronous programs behind ``next(obs)`` (worker thread)
* ``firewall.py``    what may reach the policy
* ``estimate.py``    record <-> WorldState <-> twin layout
* ``perception_service.py``  dojo.perception behind a socket (it needs another python)
* ``eval.py``        the proxy dev-eval driver (modes gt / perception), ``scripts/dojo/integ_eval.sh``
"""
