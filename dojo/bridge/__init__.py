"""dojo.bridge: our full system (runtime / agent + skills + perception + twin) behind robot-harness's strict robot
server, i.e. behind the OFFICIAL RoboDojo eval client (XPolicyLab policy ``harness_strict``).

* ``client.py``       the robot_client_v2 JSON-lines protocol (Unix socket, one request per connection)
* ``contract.py``     strict mode: the reply whitelist -> the runtime's contract observation; banned keys fail closed
* ``backend.py``      ``RobotServerBackend``: observe() + the skills Env protocol; joint action lists via ``joint_steps``
* ``perception.py``   perception routes: local service (default) or the J1 server's ``perceive`` (record converter)
* ``twin.py``         the twin service on our side, for the instruction-derived scene
* ``driver.py``       one episode: reset -> policy.reset/next <-> backend -> report (``python -m dojo.bridge.driver``)
* ``fake_server.py``  a CPU fake of the strict J1 server on the kinematic env (tests, dry runs)

Runbook: ``docs/BRIDGE-RUNBOOK.md``.
"""
