"""Local state and fail-closed exclusion of sealed measurement rows."""
import argparse
import json
from pathlib import Path


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dir', type=Path, default=Path.home() / 'dojo-board')
    repo = next((p for p in Path(__file__).resolve().parents
                 if (p / 'dojo/scoreboard/board.py').is_file()), Path.cwd())
    parser.add_argument('--repo', type=Path, default=repo)
    parser.add_argument('--main-ref', default='origin/main',
                        help='Git ref for historical drift checks; pin for exact replay')
    args = parser.parse_args()
    args.state_dir = args.state_dir.expanduser().resolve()
    args.repo = args.repo.expanduser().resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    return args


def load_rows(path, optional=False):
    if optional and not path.exists():
        import warnings
        warnings.warn(f'Missing historical input {path.name}; omitted, not imputed')
        return []
    rows = json.loads(path.read_text())
    return [r for r in rows if isinstance(r.get('layout_id', r.get('seed')), int)
            and r.get('layout_id', r.get('seed')) < 100100]
