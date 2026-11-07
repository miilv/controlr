"""G3 wrapper counting every native IK query, including speculative candidates."""
import json

from dojo.skills import g3
from dojo.skills.kinematics import X5Kin


def main():
    original_ik, original_episode = X5Kin.ik, g3.run_episode
    counts = {'queries': 0, 'failed': 0}

    def ik(self, *args, **kwargs):
        r = original_ik(self, *args, **kwargs)
        counts['queries'] += 1
        counts['failed'] += not r.ok
        return r

    def measured(*args, **kwargs):
        counts.update(queries=0, failed=0)
        result = original_episode(*args, **kwargs)
        print('[egg2-ik]', json.dumps(counts), flush=True)
        return result

    X5Kin.ik, g3.run_episode = ik, measured
    try:
        g3.main()
    finally:
        X5Kin.ik, g3.run_episode = original_ik, original_episode


if __name__ == '__main__':
    main()
