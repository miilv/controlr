"""Single-fixed-system audit (b2): the colour map in B1's appearance cost was a per-task switch (``color_tasks``).
``EstimatorConfig.color_map`` "all" applies it to every task (the uniform setting, chosen on the frame-0 sets: journal
2026-10-28-legality-uniform-color); "tasks" (the old per-task behaviour) stays the default until the lead decides."""

import pytest


def test_the_uniform_setting_applies_the_map_to_every_task_and_the_default_is_unchanged():
    pytest.importorskip("cv2")
    from dojo.perception.inverse_layout import EstimatorConfig, colour_map_for
    M = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    d = EstimatorConfig(color_M=M)
    assert d.color_map == "tasks"
    assert colour_map_for("stack_blocks", d) == M and colour_map_for("pack_objects_into_box_random", d) is None
    u = EstimatorConfig(color_M=M, color_map="all")
    assert all(colour_map_for(t, u) == M for t in ("stack_blocks", "pack_objects_into_box_random", "make_kong", "x"))
    n = EstimatorConfig(color_M=M, color_map="none")
    assert colour_map_for("stack_blocks", n) is None
    with pytest.raises(ValueError):
        colour_map_for("stack_blocks", EstimatorConfig(color_M=M, color_map="some"))
