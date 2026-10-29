"""copy_demonstration's one-to-one twins (Skills.imitate_pairing "one_to_one", default "greedy")."""
from types import SimpleNamespace

from dojo.skills.interactive import twin_assignment


def _w(objs):
    return SimpleNamespace(objects=objs, obj=lambda l: objs[l])


def _o(key):
    return SimpleNamespace(asset=SimpleNamespace(key=key), category=key.split("/")[1])


def test_one_misread_per_side_still_pairs_every_twin():
    """5.1 frame 0 of 100002: our toy car (t2) read as garage/00015, the demonstrator's watch (aim3) as an action
    camera. Greedy first-free matching takes a garage twice and stops; the assignment pairs all five."""
    objs = {"aim0": _o("Rigid/garage/00014"), "aim1": _o("Rigid/action_camera/00000"), "aim2": _o("Rigid/toy_car/00001"),
            "aim3": _o("Rigid/action_camera/00002"), "aim4": _o("Rigid/phone/00000"),
            "t0": _o("Rigid/garage/00014"), "t1": _o("Rigid/action_camera/00000"), "t2": _o("Rigid/garage/00015"),
            "t3": _o("Rigid/watch/00000"), "t4": _o("Rigid/phone/00000")}
    tw = twin_assignment(_w(objs), [f"aim{i}" for i in range(5)], [f"t{i}" for i in range(5)])
    assert tw == {f"aim{i}": f"t{i}" for i in range(5)}


def test_missing_objects_are_skipped():
    objs = {"aim0": _o("Rigid/watch/00000"), "t0": _o("Rigid/watch/00000"), "t1": _o("Rigid/pen/00000")}
    assert twin_assignment(_w(objs), ["aim0", "aim9"], ["t0", "t1"]) == {"aim0": "t0"}
    assert twin_assignment(_w(objs), [], ["t0"]) == {}
