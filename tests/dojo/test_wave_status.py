import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('wave_status', Path(__file__).parents[2] / 'scripts/dojo/queues/wave_status.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_retry_and_other_jobs():
    schedule = {'pairs': [{'family': 'waist', 'order': ['c4b_a', 'c4b_b']}]}
    rows = m.summarize(schedule, 'start c4b_a\ndone c4b_a rc=75\nstart c4b_a__retry\ndone c4b_a__retry rc=0\nstart c4b_b\ndone c4b_unrelated rc=0')
    assert rows[0]['done'] == 1
    assert rows[0]['running'] == 1
    assert rows[0]['rc'] == {'0': 1}
    assert rows[0]['eta_hours'] == 0.54


def test_lane_admission_fails_closed(tmp_path):
    import subprocess
    lane = (Path(__file__).parents[2] / 'scripts/dojo/queues/lane2.sh').read_text()
    cap = lane[lane.index('cap_ok()'):lane.index('# The sole allowed coordination file.')]
    (tmp_path / 'lane2.caps').write_text('2 5\n')
    base = f'''S={tmp_path}
others() {{ echo 0; }}
mine() {{ echo 0; }}
sides() {{ echo 0; }}
vram_free() {{ echo 20000; }}
nvidia-smi() {{ echo "$USED"; }}
{cap}
cap_ok
'''
    for used, ok in [('16384', True), ('16385', False), ('', False), ('error', False)]:
        result = subprocess.run(['bash', '-c', 'USED='+repr(used)+'\n'+base])
        assert (result.returncode == 0) == ok


def test_boot_stamp_only_authorized_coord_file(tmp_path):
    import subprocess
    lane = (Path(__file__).parents[2] / 'scripts/dojo/queues/lane2.sh').read_text()
    gate = lane[lane.index('boot_gate()'):lane.index('take_lock()')]
    (tmp_path / 'coord').mkdir()
    stamp = tmp_path / 'coord/lane_gate.stamp'
    stamp.write_text('9999999999 future other\n')
    # A future or recent stamp waits; STOP exits before overwriting it.
    script = f'HOME={tmp_path}\nlog() {{ :; }}\nstopped() {{ return 0; }}\n{gate}\nboot_gate'
    assert subprocess.run(['bash', '-c', script]).returncode == 1
    assert stamp.read_text() == '9999999999 future other\n'
    stamp.write_text('1 old other\n')
    assert subprocess.run(['bash', '-c', script]).returncode == 0
    assert stamp.read_text().split()[-1] == 'ilia-ab'
