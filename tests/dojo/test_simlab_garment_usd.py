"""The physics-only twin writes a garment's particles back to USD (RoboDojo reads them from there) for configs that
spawn a Garment, and only for those (CPU only: no Isaac)."""

from __future__ import annotations


def test_config_spawns_garment_reads_the_object_type(tmp_path, monkeypatch):
    from dojo.simlab import paths, runner
    (tmp_path / "fold_x.yml").write_text("Garment:\n  -\n    common:\n      xlim: [-0.02, 0.02]\n")
    (tmp_path / "stack_x.yml").write_text("Rigid:\n  -\n    common:\n      xlim: [0, 1]\n# Garment: mentioned only\n")
    monkeypatch.setattr(paths, "TASK_CONFIGS", tmp_path)
    assert runner.config_spawns_garment("fold_x")
    assert not runner.config_spawns_garment("stack_x")
    assert not runner.config_spawns_garment("missing_task")


def test_physics_only_kit_settings(monkeypatch):
    from dojo.simlab import runner
    monkeypatch.delenv("SIMLAB_PHYSICS_KIT_ARGS", raising=False)
    assert runner.physics_only_kit_settings() == runner.PHYSICS_ONLY_SETTINGS
    s = runner.physics_only_kit_settings(runner.GARMENT_KIT_SETTINGS)
    assert s[:len(runner.PHYSICS_ONLY_SETTINGS)] == runner.PHYSICS_ONLY_SETTINGS
    assert "--/physics/updateParticlesToUsd=true" in s and "--/physics/updateToUsd=true" in s
    monkeypatch.setenv("SIMLAB_PHYSICS_KIT_ARGS", "--/a=1 --/b=2")
    assert runner.physics_only_kit_settings()[-2:] == ("--/a=1", "--/b=2")
