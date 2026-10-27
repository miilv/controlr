"""List tasks whose config has numbered-series labels (label_prefix) or variable instance counts, and which D0 sets
have frames for them.  python scripts/perception/sam/series_tasks.py"""
from pathlib import Path
from dojo.perception.inverse_layout import task_map
H = Path.home()
d0sets = {"d0_51": H / "dojo-data/d0_51", "d0_ext": H / "dojo-data/d0_ext", "d0": H / "dojo-data/d0",
          "coin": H / "dojo-data/pmodels/d0_coin", "sweep51": H / "dojo-data/sam/sweep51", "laptop51": H / "dojo-data/sam/laptop51"}
tm = task_map()
for t in sorted(tm):
    rows = []
    for lab, d in tm[t]["labels"].items():
        sm = d.get("select_mode") or {}
        pre = sm.get("label_prefix")
        inn, cn, nums = sm.get("select_instance_nums"), sm.get("select_category_nums"), sm.get("nums")
        var = any(isinstance(x, list) and len(x) == 2 and x[0] != x[1] for x in (inn, cn, nums) if x)
        if pre or var:
            rows.append(f"{lab}: mode={sm.get('mode')} prefix={pre} inst={inn} cat={cn} nums={nums} n_labels={len(sm.get('label') or [])}")
    if rows:
        have = [k for k, p in d0sets.items() if (p / t).is_dir()]
        print(t, "| D0:", have or "-")
        for r in rows:
            print("   ", r)
