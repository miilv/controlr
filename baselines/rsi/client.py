"""Official-client suffix evaluation without renumbering frozen input positions.

The existing dev_eval layout substitution remains unchanged. Mark only the
prefix positions complete, so native SeedManager draws stay at positions 5–9.
"""
from __future__ import annotations


def exclude_prefix(seed_manager_class, start: int) -> None:
    if start != 5:
        raise ValueError('suffix pass starts at position 5')
    original = seed_manager_class.init_eval

    def init(self, completed_layout_ids=None, abandoned_layout_ids=None):
        completed = set(map(int, completed_layout_ids or [])) | set(range(start))
        original(self, completed_layout_ids=sorted(completed), abandoned_layout_ids=abandoned_layout_ids)

    seed_manager_class.init_eval = init


def main() -> None:
    from dojo.simlab import dev_eval
    original = dev_eval.patch_seed_manager

    def patch(root):
        original(root)
        from env.seed_manager import seed_manager as sm
        exclude_prefix(sm.SeedManager, 5)

    dev_eval.patch_seed_manager = patch
    dev_eval.main()


if __name__ == '__main__':
    main()
