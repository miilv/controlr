"""match_and_pick_from_conveyor: a pick's wrist look that finds nothing at the head track aborts the pick
(``belt_wrist_veto``, default off; compute4 camera rows: every pick after such a look missed, 8 / 8, no successful pick
had one, 0 / 3)."""


def test_wrist_look_absent_rule():
    from dojo.skills.interactive import wrist_look_absent
    assert wrist_look_absent({"ok": False, "iou": 0.0012, "visible_frac": 1.0})
    assert wrist_look_absent({"ok": False, "iou": 0.0, "visible_frac": 0.0})
    assert not wrist_look_absent({"ok": True, "iou": 0.83})
    assert not wrist_look_absent({"ok": False, "iou": 0.70})                       # a fit, just not accepted
    assert not wrist_look_absent({"ok": False, "iou": 0.0, "p_why": "truncated: prior kept"})   # overfills the view
    assert not wrist_look_absent({"ok": False, "iou": 0.0, "why": "perception not ok (truncated: prior kept, no re-hover)"})


def test_belt_wrist_veto_is_off_by_default():
    from dojo.runtime.policy import RuntimeConfig
    assert RuntimeConfig().belt_wrist_veto == "off"
