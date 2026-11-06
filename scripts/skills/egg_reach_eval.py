"""G3 diagnostic wrapper: count every native IK query, including speculative candidates."""
import json
from dojo.skills import g3
from dojo.skills.kinematics import X5Kin
original = X5Kin.ik
counts = {'queries': 0, 'failed': 0}
def ik(self, *args, **kwargs):
    r = original(self, *args, **kwargs)
    counts['queries'] += 1
    counts['failed'] += not r.ok
    return r
X5Kin.ik = ik
run_episode = g3.run_episode
def measured(*args, **kwargs):
    counts.update(queries=0, failed=0)
    result = run_episode(*args, **kwargs)
    print('[egg2-ik]', json.dumps(counts), flush=True)
    return result
g3.run_episode = measured
g3.main()
