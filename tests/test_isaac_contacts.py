"""Force-stop rules of the Isaac server (controlr.robot.isaac.contacts), on CPU."""

from __future__ import annotations

import numpy as np

from controlr.robot.isaac import contacts as C

LIM = C.StopLimits(env_n=80.0, box_n=30.0, object_n=40.0, held_object_n=40.0)


def test_pair_forces_take_the_max_body_per_group():
    m = np.zeros((3, 3, 3))
    m[0, 1] = [0, 30, 40]          # arm body 0 vs box: 50 N
    m[1, 1] = [0, 0, 20]           # arm body 1 vs box: 20 N
    m[2, 2] = [3, 4, 0]            # pad vs object: 5 N
    f = C.pair_forces(m, ["arm", "arm", "gripper"], ["table", "box", "object"])
    assert f == {"arm-box": 50.0, "gripper-object": 5.0}
    assert C.pair_forces(np.full((1, 1, 3), np.nan), ["arm"], ["box"]) == {"unstable": float("inf")}
    pk = C.pair_forces(np.array([[[0, 0, 12.0], [0, 0, 0]]]), ["packet"], ["box_wall", "table"], prefix="packet")
    assert pk == {"packet-box_wall": 12.0}


def test_groups_and_parts_from_paths():
    assert C.env_group("/World/Bin/Front", "/World/Waffle") == "box"
    assert C.env_group("/World/Waffle", "/World/Waffle") == "object"
    assert C.env_group("/World/Mat/Base", "/World/Waffle") == "table"
    assert C.packet_part("/World/Bin/Bottom") == "box_floor" and C.packet_part("/World/Bin/Back") == "box_wall"
    assert C.packet_part("/World/Bench/Slab") == "table"


def test_held_packet_grip_forces_never_stop_but_a_wall_jam_does():
    """Live rotation round: the grip's own pad forces reached 85-89 N in free air (false
    STOPs); genuine wall jams were 47-222 N. Only packet-vs-environment force counts now."""
    floors = C.stop_floors({"gripper-object": 14.0}, LIM)
    free_air = {"gripper-object": 89.0, "arm-object": 0.5}
    assert C.check_stop(free_air, floors, LIM, held=True, gripping=False) is None
    jam = {"gripper-object": 60.0, "packet-box_wall": 47.0}
    hit = C.check_stop(jam, floors, LIM, held=True, gripping=False)
    assert hit["kind"] == "held_object" and hit["pair"] == "packet-box_wall"
    # not held: pressing the packet with the fingers stops at the object limit
    hit = C.check_stop({"gripper-object": 45.0}, {}, LIM, held=False, gripping=False)
    assert hit["kind"] == "object"
    assert C.check_stop({"gripper-object": 45.0}, {}, LIM, held=False, gripping=True) is None
    # closing on the packet loads its contact with the mat: not a jam
    assert C.check_stop({"packet-table": 60.0}, {}, LIM, held=True, gripping=True) is None
    # the held packet's contacts are not extrapolated (they jitter with the grip)
    assert C.check_stop({"packet-table": 22.0}, {}, LIM, held=True, gripping=False,
                        prev={"packet-table": 4.0}, predict=True) is None
    # a packet resting on the mat (not held) never stops the arm
    assert C.check_stop({"packet-table": 300.0}, {}, LIM, held=False, gripping=False) is None


def test_box_touch_stops_at_the_box_limit_and_a_resting_contact_can_back_away():
    assert C.check_stop({"gripper-box": 25.0}, {}, LIM, held=False, gripping=False) is None
    assert C.check_stop({"gripper-box": 31.0}, {}, LIM, held=False, gripping=False)["kind"] == "env"
    assert C.check_stop({"arm-table": 70.0}, {}, LIM, held=False, gripping=False) is None
    # already pressing 25 N on the box at the start of the motion: +20 N allowed (cap 2x)
    floors = C.stop_floors({"gripper-box": 25.0}, LIM)
    assert floors["gripper-box"] == 45.0
    assert C.check_stop({"gripper-box": 40.0}, floors, LIM, held=False, gripping=False) is None
    assert C.stop_floors({"gripper-box": 100.0}, LIM)["gripper-box"] == 60.0


def test_pushing_the_light_box_stops_even_below_the_force_limit():
    """Measured (contacts-and-speed): the 0.4 kg box slides at a few newtons, and at 5 ms
    sampling a push of 160 mm never crossed 30 N. Contact + the box moved > 5 mm stops."""
    push = {"gripper-box": 4.0}
    assert C.check_stop(push, {}, LIM, held=False, gripping=False, box_moved_m=0.003) is None
    hit = C.check_stop(push, {}, LIM, held=False, gripping=False, box_moved_m=0.008)
    assert hit["kind"] == "env" and hit["pushed"] and hit["moved_m"] == 0.008
    # the box moved but nothing of the robot touches it (e.g. the packet was dropped into it)
    assert C.check_stop({"packet-box_floor": 3.0}, {}, LIM, held=False, gripping=False, box_moved_m=0.02) is None
    hit = C.check_stop({"packet-box_wall": 3.0}, {}, LIM, held=True, gripping=False, box_moved_m=0.02)
    assert hit["kind"] == "held_object" and hit["pushed"]
    off = C.StopLimits(box_push_m=0.0)
    assert C.check_stop(push, {}, off, held=False, gripping=False, box_moved_m=0.5) is None


def test_unstable_and_predictive_stop():
    assert C.check_stop({"arm-box": 1e7}, {}, LIM, held=False, gripping=False)["kind"] == "unstable"
    assert C.check_stop({"unstable": float("inf")}, {}, LIM, held=False, gripping=False)["kind"] == "unstable"
    # rising 18 -> 26 N: 26 < 30, but one more sample ahead would be 34 N
    hit = C.check_stop({"gripper-box": 26.0}, {}, LIM, held=False, gripping=False,
                       prev={"gripper-box": 18.0}, predict=True)
    assert hit["kind"] == "env" and hit["predicted"]
    assert C.check_stop({"gripper-box": 26.0}, {}, LIM, held=False, gripping=False,
                        prev={"gripper-box": 18.0}) is None
