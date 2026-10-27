"""Kit log watch: infrastructure failures (uncooked SDF colliders) are seen per episode, not counted as skill failures."""

from dojo.simlab.kitlog import KitLogWatch
from dojo.skills.g3 import write_summary

COOK = ("2026-10-27T03:01:59Z [7,265ms] [Warning] [omni.physx.cooking.plugin] UjitsoMeshCookingContext: cooking failure "
        "for /World/envs/env_0/rigid/factory_nut/factory_nut_2_5/collision/collisions\n")


def test_watch_reports_only_lines_after_its_mark(tmp_path):
    log = tmp_path / "kit.log"
    log.write_text("start\n" + COOK)
    w = KitLogWatch(str(log))                    # an earlier episode's failure is before the mark
    assert w.since() == {}
    with log.open("a") as f:
        f.write("[Info] load\n" + COOK + COOK)
    got = w.since()
    assert got["cooking_failures"] == 2 and "factory_nut_2_5" in got["first"]
    w.mark()
    assert w.since() == {}


def test_watch_outside_kit_is_silent(tmp_path):
    assert KitLogWatch(None).since() == {}
    assert KitLogWatch(str(tmp_path / "missing.log")).since() == {}


def test_summary_leaves_infra_failures_out(tmp_path):
    rows = [{"seed": 1, "stable": True, "success": True, "score": 1.0, "steps": 100, "step_lim": 500, "wall_s": 1.0,
             "plan_s": 1.0},
            {"seed": 2, "stable": True, "success": False, "score": 0.0, "steps": 50, "step_lim": 500, "wall_s": 1.0,
             "plan_s": 1.0, "category": "infra_failure", "infra_failure": {"cooking_failures": 3}}]
    s = write_summary(tmp_path, "fasten_screws", rows)
    assert s["n"] == 1 and s["success"] == 1 and s["infra_failures"] == 1 and s["infra_seeds"] == [2]
    assert s["unstable"] == 0 and "infra_failure" not in s["categories"]
